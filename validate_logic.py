#!/usr/bin/env python3
"""
Comprehensive logical validation of simulation results.
Checks data integrity, business logic, and cross-file consistency.
"""

import pandas as pd
import sys
from pathlib import Path

def validate_simulation_logic():
    """Run all validation checks."""
    
    print('='*70)
    print('LOGICAL CONSISTENCY VALIDATION')
    print('='*70)
    
    # Load all CSV files
    results_dir = Path('simulation_results')
    bookings = pd.read_csv(results_dir / 'bookings.csv')
    requests = pd.read_csv(results_dir / 'booking_requests.csv')
    dtd = pd.read_csv(results_dir / 'dtd_analysis.csv')
    segments = pd.read_csv(results_dir / 'segment_analysis.csv')
    flights = pd.read_csv(results_dir / 'flight_metrics.csv')
    
    errors = []
    warnings = []
    
    print('\n1. DATA INTEGRITY CHECKS')
    print('-'*70)
    
    # Check 1: Booking count consistency
    booking_count = len(bookings)
    accepted_count = len(requests[requests['accepted'] == True])
    print(f'✓ Bookings CSV: {booking_count} records')
    print(f'✓ Accepted requests: {accepted_count} records')
    if booking_count == accepted_count:
        print(f'  ✓ MATCH')
    else:
        error = f'MISMATCH: {booking_count} bookings vs {accepted_count} accepted requests'
        print(f'  ✗ {error}')
        errors.append(error)
    
    # Check 2: Revenue consistency
    total_revenue_bookings = bookings['total_paid'].sum()
    print(f'\n✓ Total revenue (bookings.csv): ${total_revenue_bookings:,.2f}')
    
    # Check 3: Request totals
    total_requests = len(requests)
    accepted = len(requests[requests['accepted'] == True])
    rejected = len(requests[requests['accepted'] == False])
    print(f'\n✓ Total requests: {total_requests}')
    print(f'  - Accepted: {accepted} ({accepted/total_requests*100:.1f}%)')
    print(f'  - Rejected: {rejected} ({rejected/total_requests*100:.1f}%)')
    if accepted + rejected == total_requests:
        print(f'  ✓ Sum check: PASS')
    else:
        error = f'Request sum mismatch: {accepted} + {rejected} != {total_requests}'
        print(f'  ✗ Sum check: FAIL - {error}')
        errors.append(error)
    
    # Check 4: DTD analysis sum
    dtd_total_bookings = dtd['booking_count'].sum()
    print(f'\n✓ DTD analysis total bookings: {dtd_total_bookings}')
    if dtd_total_bookings == booking_count:
        print(f'  ✓ Match with bookings')
    else:
        error = f'DTD booking count mismatch: {dtd_total_bookings} vs {booking_count}'
        print(f'  ✗ {error}')
        errors.append(error)
    
    dtd_total_revenue = dtd['total_revenue'].sum()
    print(f'✓ DTD analysis total revenue: ${dtd_total_revenue:,.2f}')
    if abs(dtd_total_revenue - total_revenue_bookings) < 1:
        print(f'  ✓ Match with bookings')
    else:
        error = f'DTD revenue mismatch: ${dtd_total_revenue:,.2f} vs ${total_revenue_bookings:,.2f}'
        print(f'  ✗ {error}')
        errors.append(error)
    
    # Check 5: Segment analysis sum
    segment_total_bookings = segments['booking_count'].sum()
    segment_total_revenue = segments['total_revenue'].sum()
    print(f'\n✓ Segment analysis total bookings: {segment_total_bookings}')
    if segment_total_bookings == booking_count:
        print(f'  ✓ Match with bookings')
    else:
        error = f'Segment booking count mismatch: {segment_total_bookings} vs {booking_count}'
        print(f'  ✗ {error}')
        errors.append(error)
    
    print(f'✓ Segment analysis total revenue: ${segment_total_revenue:,.2f}')
    if abs(segment_total_revenue - total_revenue_bookings) < 1:
        print(f'  ✓ Match with bookings')
    else:
        error = f'Segment revenue mismatch: ${segment_total_revenue:,.2f} vs ${total_revenue_bookings:,.2f}'
        print(f'  ✗ {error}')
        errors.append(error)
    
    # Check 6: Flight metrics sum
    flight_total_bookings = flights['bookings'].sum()
    flight_total_revenue = flights['revenue'].sum()
    print(f'\n✓ Flight metrics total bookings: {flight_total_bookings}')
    if flight_total_bookings == booking_count:
        print(f'  ✓ Match with bookings')
    else:
        warning = f'Flight booking count mismatch: {flight_total_bookings} vs {booking_count}'
        print(f'  ⚠ {warning}')
        warnings.append(warning)
    
    print(f'✓ Flight metrics total revenue: ${flight_total_revenue:,.2f}')
    if abs(flight_total_revenue - total_revenue_bookings) < 1:
        print(f'  ✓ Match with bookings')
    else:
        warning = f'Flight revenue mismatch: ${flight_total_revenue:,.2f} vs ${total_revenue_bookings:,.2f}'
        print(f'  ⚠ {warning}')
        warnings.append(warning)
    
    print('\n2. BUSINESS LOGIC CHECKS')
    print('-'*70)
    
    # Check 7: Cancellation consistency
    cancelled_bookings = len(bookings[bookings['cancelled'] == True])
    print(f'✓ Cancelled bookings: {cancelled_bookings}')
    print(f'✓ Cancellation rate: {cancelled_bookings/booking_count*100:.1f}%')
    
    # Check 8: Party size validation
    invalid_party_sizes = len(bookings[bookings['party_size'] <= 0])
    print(f'\n✓ Invalid party sizes (<=0): {invalid_party_sizes}')
    if invalid_party_sizes == 0:
        print(f'  ✓ Status: PASS')
    else:
        error = f'Found {invalid_party_sizes} bookings with invalid party size'
        print(f'  ✗ Status: FAIL - {error}')
        errors.append(error)
    
    # Check 9: Revenue validation
    negative_revenue = len(bookings[bookings['total_paid'] < 0])
    zero_revenue = len(bookings[bookings['total_paid'] == 0])
    print(f'\n✓ Negative revenue bookings: {negative_revenue}')
    print(f'✓ Zero revenue bookings: {zero_revenue}')
    if negative_revenue == 0:
        print(f'  ✓ Status: PASS')
    else:
        error = f'Found {negative_revenue} bookings with negative revenue'
        print(f'  ✗ Status: FAIL - {error}')
        errors.append(error)
    
    # Check 10: Days to departure validation
    negative_dtd = len(bookings[bookings['days_to_departure'] < 0])
    print(f'\n✓ Negative DTD bookings: {negative_dtd}')
    if negative_dtd == 0:
        print(f'  ✓ Status: PASS')
    else:
        error = f'Found {negative_dtd} bookings with negative days-to-departure'
        print(f'  ✗ Status: FAIL - {error}')
        errors.append(error)
    
    # Check 11: Cabin class distribution
    cabin_counts = bookings['cabin_class'].value_counts()
    print(f'\n✓ Cabin class distribution:')
    for cabin, count in cabin_counts.items():
        print(f'  - {cabin}: {count} ({count/booking_count*100:.1f}%)')
    
    # Check 12: Segment distribution
    segment_counts = bookings['customer_segment'].value_counts()
    print(f'\n✓ Customer segment distribution:')
    for seg, count in segment_counts.items():
        print(f'  - {seg}: {count} ({count/booking_count*100:.1f}%)')
    
    # Check 13: Load factor validation
    print(f'\n✓ Flight load factors:')
    print(f'  - Min: {flights["load_factor"].min():.1f}%')
    print(f'  - Max: {flights["load_factor"].max():.1f}%')
    print(f'  - Avg: {flights["load_factor"].mean():.1f}%')
    over_100 = len(flights[flights['load_factor'] > 100])
    print(f'  - Flights over 100%: {over_100} (overbooking)')
    
    over_120 = len(flights[flights['load_factor'] > 120])
    if over_120 > 0:
        warning = f'{over_120} flights have load factor > 120% (excessive overbooking)'
        print(f'  ⚠ WARNING: {warning}')
        warnings.append(warning)
    
    # Check 14: Rejection reason validation
    rejection_reasons = requests[requests['accepted'] == False]['rejection_reason'].value_counts()
    print(f'\n✓ Rejection reasons:')
    for reason, count in rejection_reasons.items():
        print(f'  - {reason}: {count} ({count/rejected*100:.1f}%)')
    
    # Check 15: Revenue per passenger validation
    bookings['revenue_per_pax'] = bookings['total_paid'] / bookings['party_size']
    print(f'\n✓ Revenue per passenger:')
    print(f'  - Min: ${bookings["revenue_per_pax"].min():.2f}')
    print(f'  - Max: ${bookings["revenue_per_pax"].max():.2f}')
    print(f'  - Avg: ${bookings["revenue_per_pax"].mean():.2f}')
    
    # Check 16: Date logic validation
    print(f'\n✓ Date consistency:')
    bookings['booking_date'] = pd.to_datetime(bookings['booking_time'])
    bookings['departure_date_parsed'] = pd.to_datetime(bookings['departure_date'])
    future_bookings = len(bookings[bookings['booking_date'] > bookings['departure_date_parsed']])
    if future_bookings == 0:
        print(f'  ✓ All bookings made before departure')
    else:
        error = f'Found {future_bookings} bookings made after departure date'
        print(f'  ✗ {error}')
        errors.append(error)
    
    # Check 17: WTP vs Price validation
    print(f'\n✓ Price vs WTP consistency:')
    bookings['avg_fare'] = bookings['total_paid'] / bookings['party_size']
    exceeds_wtp = len(bookings[bookings['avg_fare'] > bookings['willingness_to_pay']])
    if exceeds_wtp == 0:
        print(f'  ✓ All bookings: fare <= WTP')
    else:
        warning = f'{exceeds_wtp} bookings have fare > WTP (possible choice model behavior)'
        print(f'  ⚠ {warning}')
        warnings.append(warning)
    
    # Check 18: Cancellation date logic
    cancelled = bookings[bookings['cancelled'] == True].copy()
    if len(cancelled) > 0:
        cancelled['cancellation_date_parsed'] = pd.to_datetime(cancelled['cancellation_time'])
        invalid_cancel = len(cancelled[cancelled['cancellation_date_parsed'] < cancelled['booking_date']])
        if invalid_cancel == 0:
            print(f'\n✓ Cancellation dates: All after booking date')
        else:
            error = f'Found {invalid_cancel} cancellations before booking date'
            print(f'\n✗ {error}')
            errors.append(error)
    
    # Summary
    print('\n' + '='*70)
    print('VALIDATION SUMMARY')
    print('='*70)
    
    if len(errors) == 0 and len(warnings) == 0:
        print('✓ ALL CHECKS PASSED - Logic is correct!')
        return 0
    else:
        if len(errors) > 0:
            print(f'\n✗ ERRORS FOUND: {len(errors)}')
            for i, error in enumerate(errors, 1):
                print(f'  {i}. {error}')
        
        if len(warnings) > 0:
            print(f'\n⚠ WARNINGS: {len(warnings)}')
            for i, warning in enumerate(warnings, 1):
                print(f'  {i}. {warning}')
        
        return 1 if len(errors) > 0 else 0


if __name__ == '__main__':
    sys.exit(validate_simulation_logic())
