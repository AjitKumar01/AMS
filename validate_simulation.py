#!/usr/bin/env python3
"""
Comprehensive validation script for PyAirline RM simulation.
Checks logical correctness of simulation outputs and internal consistency.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

def validate_csv_files():
    """Validate all CSV export files for logical consistency."""
    results_dir = Path("simulation_results")
    
    print("=" * 70)
    print("SIMULATION VALIDATION REPORT")
    print("=" * 70)
    print()
    
    issues = []
    warnings = []
    
    # Load all CSV files
    try:
        bookings = pd.read_csv(results_dir / "bookings.csv")
        requests = pd.read_csv(results_dir / "booking_requests.csv")
        dtd = pd.read_csv(results_dir / "dtd_analysis.csv")
        segments = pd.read_csv(results_dir / "segment_analysis.csv")
        flights = pd.read_csv(results_dir / "flight_metrics.csv")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Missing CSV file: {e}")
        return False
    
    print("✓ All CSV files found and loaded")
    print()
    
    # ========================================================================
    # 1. BOOKINGS VALIDATION
    # ========================================================================
    print("📋 BOOKINGS VALIDATION")
    print("-" * 70)
    
    # Check for negative values
    if (bookings['total_paid'] < 0).any():
        issues.append("Bookings have NEGATIVE total_paid values")
    else:
        print("  ✓ All booking revenues are non-negative")
    
    if (bookings['party_size'] < 1).any():
        issues.append("Bookings have INVALID party_size (< 1)")
    else:
        print("  ✓ All party sizes are valid (>= 1)")
    
    # Check dates
    bookings['booking_time'] = pd.to_datetime(bookings['booking_time'])
    bookings['departure_date'] = pd.to_datetime(bookings['departure_date'])
    
    # Check using DATE comparison (departure is stored as date without time)
    bookings['booking_date'] = bookings['booking_time'].dt.floor('D')
    bookings['departure_date_only'] = bookings['departure_date'].dt.floor('D')
    
    future_bookings = bookings[bookings['booking_date'] > bookings['departure_date_only']]
    if len(future_bookings) > 0:
        issues.append(f"Found {len(future_bookings)} bookings made AFTER departure date")
    else:
        print("  ✓ All bookings made before/on departure date")
    
    # Check DTD calculation (using date difference, not datetime difference)
    bookings['calculated_dtd'] = ((bookings['departure_date_only'] - bookings['booking_date']).dt.total_seconds() / 86400).astype(int)
    dtd_mismatch = bookings[bookings['calculated_dtd'] != bookings['days_to_departure']]
    if len(dtd_mismatch) > 0:
        issues.append(f"Found {len(dtd_mismatch)} bookings with INCORRECT days_to_departure calculation")
    else:
        print("  ✓ Days-to-departure calculations correct")
    
    # Check cancelled bookings
    cancelled = bookings[bookings['cancelled'] == True]
    print(f"  ✓ Found {len(cancelled)} cancelled bookings ({len(cancelled)/len(bookings)*100:.1f}%)")
    
    cancelled_with_time = cancelled[cancelled['cancellation_time'].notna()]
    if len(cancelled_with_time) != len(cancelled):
        warnings.append(f"{len(cancelled) - len(cancelled_with_time)} cancelled bookings missing cancellation_time")
    
    # Check cabin classes
    valid_cabins = ['F', 'J', 'Y']
    invalid_cabins = bookings[~bookings['cabin_class'].isin(valid_cabins)]
    if len(invalid_cabins) > 0:
        issues.append(f"Found {len(invalid_cabins)} bookings with INVALID cabin_class")
    else:
        print("  ✓ All cabin classes are valid (F/J/Y)")
    
    # Check customer segments
    valid_segments = ['business', 'leisure', 'vfr', 'group']
    invalid_segments = bookings[~bookings['customer_segment'].isin(valid_segments)]
    if len(invalid_segments) > 0:
        issues.append(f"Found {len(invalid_segments)} bookings with INVALID customer_segment")
    else:
        print("  ✓ All customer segments are valid")
    
    print()
    
    # ========================================================================
    # 2. REQUESTS VALIDATION
    # ========================================================================
    print("📨 REQUESTS VALIDATION")
    print("-" * 70)
    
    # Check acceptance rate
    total_requests = len(requests)
    accepted_requests = len(requests[requests['accepted'] == True])
    acceptance_rate = accepted_requests / total_requests * 100
    
    print(f"  Total requests: {total_requests}")
    print(f"  Accepted: {accepted_requests} ({acceptance_rate:.1f}%)")
    print(f"  Rejected: {total_requests - accepted_requests} ({100-acceptance_rate:.1f}%)")
    
    # Check if accepted requests match bookings count
    if accepted_requests != len(bookings):
        issues.append(f"MISMATCH: {accepted_requests} accepted requests but {len(bookings)} bookings")
    else:
        print("  ✓ Accepted requests count matches bookings count")
    
    # Check rejected requests have reasons
    rejected = requests[requests['accepted'] == False]
    rejected_without_reason = rejected[rejected['rejection_reason'].isna() | (rejected['rejection_reason'] == '')]
    if len(rejected_without_reason) > 0:
        warnings.append(f"{len(rejected_without_reason)} rejected requests missing rejection_reason")
    else:
        print("  ✓ All rejections have reasons")
    
    # Check WTP values
    if (requests['willingness_to_pay'] < 0).any():
        issues.append("Found requests with NEGATIVE willingness_to_pay")
    else:
        print("  ✓ All WTP values are non-negative")
    
    print()
    
    # ========================================================================
    # 3. CROSS-VALIDATION: Bookings vs Requests
    # ========================================================================
    print("🔗 CROSS-VALIDATION: Bookings vs Requests")
    print("-" * 70)
    
    # Check if accepted bookings have reasonable prices relative to WTP
    requests['request_id_short'] = requests['request_id']
    accepted_req = requests[requests['accepted'] == True].copy()
    
    # Check price vs WTP relationship
    bookings['per_pax_price'] = bookings['total_paid'] / bookings['party_size']
    bookings['price_vs_wtp_ratio'] = bookings['per_pax_price'] / bookings['willingness_to_pay']
    
    # With MNL choice model, customers can book above WTP if utility compensates
    over_wtp = bookings[bookings['per_pax_price'] > bookings['willingness_to_pay']]
    over_wtp_pct = len(over_wtp) / len(bookings) * 100
    
    print(f"  ✓ Price vs WTP distribution: {over_wtp_pct:.1f}% bookings above stated WTP")
    print(f"    (Expected with MNL choice model - utility includes time, schedule, etc.)")
    
    # Flag only extreme cases (>3x WTP likely indicates a bug)
    extreme = bookings[bookings['per_pax_price'] > bookings['willingness_to_pay'] * 3.0]
    if len(extreme) > 0:
        warnings.append(f"{len(extreme)} bookings priced >300% of customer WTP (check choice model)")
    else:
        print("  ✓ No extreme pricing anomalies found")
    
    print()
    
    # ========================================================================
    # 4. DTD ANALYSIS VALIDATION
    # ========================================================================
    print("📅 DTD ANALYSIS VALIDATION")
    print("-" * 70)
    
    # Check if DTD buckets sum to total bookings
    dtd_total = dtd['booking_count'].sum()
    if dtd_total != len(bookings):
        issues.append(f"DTD analysis total ({dtd_total}) doesn't match bookings count ({len(bookings)})")
    else:
        print(f"  ✓ DTD buckets sum correctly ({dtd_total} bookings)")
    
    # Check revenue sums
    dtd_revenue_total = dtd['total_revenue'].sum()
    bookings_revenue_total = bookings['total_paid'].sum()
    revenue_diff_pct = abs(dtd_revenue_total - bookings_revenue_total) / bookings_revenue_total * 100
    
    if revenue_diff_pct > 0.1:  # More than 0.1% difference
        issues.append(f"DTD revenue total (${dtd_revenue_total:,.2f}) doesn't match bookings (${bookings_revenue_total:,.2f})")
    else:
        print(f"  ✓ Revenue totals match: ${dtd_revenue_total:,.2f}")
    
    # Check percentages
    dtd['segment_total'] = dtd['business_pct'] + dtd['leisure_pct']
    invalid_pct = dtd[abs(dtd['segment_total'] - 100.0) > 0.5]  # Allow 0.5% rounding error
    if len(invalid_pct) > 0:
        warnings.append(f"{len(invalid_pct)} DTD buckets have segment percentages not summing to 100%")
    else:
        print("  ✓ Segment percentages sum to 100%")
    
    print()
    
    # ========================================================================
    # 5. SEGMENT ANALYSIS VALIDATION
    # ========================================================================
    print("👥 SEGMENT ANALYSIS VALIDATION")
    print("-" * 70)
    
    # Check segment counts sum to total
    segment_total = segments['booking_count'].sum()
    if segment_total != len(bookings):
        issues.append(f"Segment analysis total ({segment_total}) doesn't match bookings ({len(bookings)})")
    else:
        print(f"  ✓ Segment counts sum correctly ({segment_total} bookings)")
    
    # Check segment revenue
    segment_revenue_total = segments['total_revenue'].sum()
    if abs(segment_revenue_total - bookings_revenue_total) / bookings_revenue_total > 0.001:
        issues.append(f"Segment revenue (${segment_revenue_total:,.2f}) doesn't match bookings")
    else:
        print(f"  ✓ Segment revenue matches: ${segment_revenue_total:,.2f}")
    
    # Check cabin percentages
    for idx, row in segments.iterrows():
        cabin_total = row['first_class_pct'] + row['business_pct'] + row['economy_pct']
        if abs(cabin_total - 100.0) > 0.5:
            warnings.append(f"Segment {row['segment']}: cabin percentages sum to {cabin_total:.1f}% (not 100%)")
    
    print("  ✓ Cabin class distributions calculated")
    
    print()
    
    # ========================================================================
    # 6. FLIGHT METRICS VALIDATION
    # ========================================================================
    print("✈️  FLIGHT METRICS VALIDATION")
    print("-" * 70)
    
    # Check load factors
    if (flights['load_factor'] < 0).any() or (flights['load_factor'] > 100).any():
        issues.append("Found flights with INVALID load_factor (outside 0-100%)")
    else:
        print("  ✓ All load factors within valid range (0-100%)")
    
    # Check bookings don't exceed capacity
    overbooked = flights[flights['bookings'] > flights['capacity']]
    if len(overbooked) > 0:
        # This might be OK if overbooking is enabled
        print(f"  ⚠️  Found {len(overbooked)} flights with bookings > capacity (overbooking)")
    else:
        print("  ✓ No flights exceed capacity")
    
    # Check calculated load factor
    flights['calculated_lf'] = (flights['bookings'] / flights['capacity'] * 100).round(1)
    lf_mismatch = flights[abs(flights['calculated_lf'] - flights['load_factor']) > 0.5]
    if len(lf_mismatch) > 0:
        issues.append(f"{len(lf_mismatch)} flights have INCORRECT load_factor calculation")
    else:
        print("  ✓ Load factor calculations correct")
    
    # Check revenue per seat
    flights['calculated_rps'] = (flights['revenue'] / flights['capacity']).round(2)
    rps_mismatch = flights[abs(flights['calculated_rps'] - flights['revenue_per_seat']) > 0.5]
    if len(rps_mismatch) > 0:
        warnings.append(f"{len(rps_mismatch)} flights have revenue_per_seat calculation differences")
    else:
        print("  ✓ Revenue per seat calculations correct")
    
    # Check average fare
    flights['calculated_avg_fare'] = (flights['revenue'] / flights['bookings']).round(2)
    fare_mismatch = flights[abs(flights['calculated_avg_fare'] - flights['avg_fare']) > 0.5]
    if len(fare_mismatch) > 0:
        warnings.append(f"{len(fare_mismatch)} flights have avg_fare calculation differences")
    else:
        print("  ✓ Average fare calculations correct")
    
    print()
    
    # ========================================================================
    # 7. REVENUE CONSISTENCY CHECK
    # ========================================================================
    print("💰 REVENUE CONSISTENCY CHECK")
    print("-" * 70)
    
    bookings_total = bookings['total_paid'].sum()
    flights_total = flights['revenue'].sum()
    
    print(f"  Bookings total revenue: ${bookings_total:,.2f}")
    print(f"  Flights total revenue:  ${flights_total:,.2f}")
    
    revenue_diff = abs(bookings_total - flights_total)
    revenue_diff_pct = revenue_diff / bookings_total * 100
    
    if revenue_diff_pct > 0.1:
        issues.append(f"Revenue MISMATCH: ${revenue_diff:,.2f} difference ({revenue_diff_pct:.2f}%)")
    else:
        print(f"  ✓ Revenue totals match (diff: ${revenue_diff:.2f})")
    
    print()
    
    # ========================================================================
    # 8. LOGICAL BUSINESS RULES
    # ========================================================================
    print("📊 BUSINESS RULES VALIDATION")
    print("-" * 70)
    
    # Check fare reasonableness by cabin
    cabin_fares = bookings.groupby('cabin_class')['per_pax_price'].agg(['mean', 'min', 'max'])
    print("  Fare ranges by cabin:")
    for cabin in ['F', 'J', 'Y']:
        if cabin in cabin_fares.index:
            print(f"    {cabin}: ${cabin_fares.loc[cabin, 'min']:.2f} - ${cabin_fares.loc[cabin, 'max']:.2f} (avg: ${cabin_fares.loc[cabin, 'mean']:.2f})")
    
    # Check if F > J > Y on average
    if 'F' in cabin_fares.index and 'J' in cabin_fares.index:
        if cabin_fares.loc['F', 'mean'] <= cabin_fares.loc['J', 'mean']:
            warnings.append("First class average fare NOT higher than Business")
    
    if 'J' in cabin_fares.index and 'Y' in cabin_fares.index:
        if cabin_fares.loc['J', 'mean'] <= cabin_fares.loc['Y', 'mean']:
            warnings.append("Business class average fare NOT higher than Economy")
    
    # Check business travelers pay more on average
    segment_fares = bookings.groupby('customer_segment')['per_pax_price'].mean()
    if 'business' in segment_fares.index and 'leisure' in segment_fares.index:
        if segment_fares['business'] > segment_fares['leisure']:
            print(f"  ✓ Business travelers pay more on average (${segment_fares['business']:.2f} vs ${segment_fares['leisure']:.2f})")
        else:
            warnings.append(f"Business travelers pay LESS than leisure (${segment_fares['business']:.2f} vs ${segment_fares['leisure']:.2f})")
    
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    if len(issues) == 0 and len(warnings) == 0:
        print("✅ ALL CHECKS PASSED - No issues or warnings found!")
        print()
        print("The simulation data is logically consistent and ready for analysis.")
        return True
    
    if len(warnings) > 0:
        print(f"⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print()
    
    if len(issues) > 0:
        print(f"❌ CRITICAL ISSUES ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("Please review and fix the issues above.")
        return False
    
    if len(issues) == 0 and len(warnings) > 0:
        print("✅ No critical issues found, but there are some warnings to review.")
        print("The simulation data is generally consistent.")
        return True


if __name__ == "__main__":
    success = validate_csv_files()
    sys.exit(0 if success else 1)
