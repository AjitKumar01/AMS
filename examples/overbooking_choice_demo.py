"""
Example demonstrating overbooking and customer choice models.

Shows:
- Overbooking optimization with no-show modeling
- Denied boarding handling
- Customer choice with Multinomial Logit (MNL)
- Buy-up and buy-down behavior
- Recapture modeling
"""

from datetime import date, timedelta, time
import numpy as np

from core.models import (
    Airport, Route, Aircraft, FlightSchedule, Customer,
    CustomerSegment, BookingClass, CabinClass, TravelSolution, Booking
)
from core.simulator import SimulationConfig, Simulator
from overbooking.optimizer import (
    OverbookingOptimizer, OverbookingMethod, NoShowModel, 
    DeniedBoardingCost
)
from choice.models import (
    MultinomialLogitModel, EnhancedChoiceModel, UtilityFunction,
    BuyUpDownModel, RecaptureModel, ChoiceSet
)


def demo_overbooking():
    """Demonstrate overbooking optimization."""
    print("=" * 70)
    print("OVERBOOKING DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create no-show model
    no_show_model = NoShowModel()
    
    # Create overbooking optimizer
    optimizer = OverbookingOptimizer(
        no_show_model=no_show_model,
        db_cost_model=DeniedBoardingCost(),
        method=OverbookingMethod.CRITICAL_FRACTILE
    )
    
    # Simulate flight with 150 capacity
    capacity = 150
    avg_fare = 400.0
    
    # Create mock bookings
    mock_bookings = []
    rng = np.random.default_rng(42)
    
    for i in range(120):
        # Mix of business and leisure
        segment = CustomerSegment.BUSINESS if i < 40 else CustomerSegment.LEISURE
        customer = Customer(
            segment=segment,
            willingness_to_pay=avg_fare * (1.2 if segment == CustomerSegment.BUSINESS else 0.8),
            advance_purchase_days=14,
            price_sensitivity=0.8 if segment == CustomerSegment.BUSINESS else 1.4
        )
        
        # Create mock booking
        solution = TravelSolution()
        solution.booking_classes = [BookingClass.Y if segment == CustomerSegment.BUSINESS else BookingClass.M]
        solution.total_price = avg_fare
        
        booking = Booking(
            customer=customer,
            solution=solution,
            party_size=1,
            total_revenue=avg_fare
        )
        mock_bookings.append(booking)
    
    # Calculate overbooking policy
    print(f"Current bookings: {len(mock_bookings)}")
    print(f"Aircraft capacity: {capacity}")
    print(f"Average fare: ${avg_fare:.2f}")
    print()
    
    policy = optimizer.calculate_overbooking_limit(
        capacity=capacity,
        current_bookings=mock_bookings,
        avg_fare=avg_fare,
        risk_tolerance=0.05
    )
    
    print("OVERBOOKING POLICY:")
    print(f"  Booking limit: {policy.booking_limit}")
    print(f"  Overbooking level: {policy.overbooking_level} seats ({policy.overbooking_rate*100:.1f}%)")
    print(f"  Expected shows: {policy.expected_shows:.1f}")
    print(f"  Risk of denied boarding: {policy.risk_of_denied_boarding*100:.1f}%")
    print(f"  Expected revenue gain: ${policy.expected_revenue_gain:.2f}")
    print(f"  Expected DB cost: ${policy.expected_db_cost:.2f}")
    print(f"  Net benefit: ${policy.net_benefit:.2f}")
    print()
    
    # Simulate show-ups
    print("SIMULATION OF SHOW-UPS:")
    n_simulations = 1000
    denied_boarding_count = 0
    total_shows = []
    
    for _ in range(n_simulations):
        shows, denied = optimizer.simulate_denied_boarding(
            mock_bookings[:policy.booking_limit],
            capacity,
            rng
        )
        total_shows.append(shows)
        if len(denied) > 0:
            denied_boarding_count += 1
    
    avg_shows = np.mean(total_shows)
    std_shows = np.std(total_shows)
    
    print(f"  Simulations run: {n_simulations}")
    print(f"  Average shows: {avg_shows:.1f} ± {std_shows:.1f}")
    print(f"  Denied boarding events: {denied_boarding_count} ({denied_boarding_count/n_simulations*100:.1f}%)")
    print(f"  Empty seats avg: {max(0, capacity - avg_shows):.1f}")
    print()
    
    # Compare with conservative approach
    print("COMPARISON WITH CONSERVATIVE APPROACH:")
    conservative_optimizer = OverbookingOptimizer(
        no_show_model=no_show_model,
        method=OverbookingMethod.RISK_AVERSE
    )
    
    conservative_policy = conservative_optimizer.calculate_overbooking_limit(
        capacity=capacity,
        current_bookings=mock_bookings,
        avg_fare=avg_fare,
        risk_tolerance=0.05
    )
    
    print(f"  Conservative booking limit: {conservative_policy.booking_limit}")
    print(f"  Conservative overbooking: {conservative_policy.overbooking_level} seats")
    print(f"  Conservative net benefit: ${conservative_policy.net_benefit:.2f}")
    print(f"  Difference: ${policy.net_benefit - conservative_policy.net_benefit:.2f}")
    print()


def demo_customer_choice():
    """Demonstrate customer choice models."""
    print("=" * 70)
    print("CUSTOMER CHOICE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create mock customer
    customer = Customer(
        segment=CustomerSegment.LEISURE,
        willingness_to_pay=500.0,
        advance_purchase_days=21,
        price_sensitivity=1.3,
        time_sensitivity=0.7,
        loyalty_score=0.3
    )
    
    print(f"Customer profile:")
    print(f"  Segment: {customer.segment.value}")
    print(f"  WTP: ${customer.willingness_to_pay:.2f}")
    print(f"  Price sensitivity: {customer.price_sensitivity:.2f}")
    print(f"  Time sensitivity: {customer.time_sensitivity:.2f}")
    print()
    
    # Create mock solutions
    jfk = Airport("JFK", "New York JFK", "New York", "USA", "America/New_York", 40.64, -73.78)
    lax = Airport("LAX", "Los Angeles Intl", "Los Angeles", "USA", "America/Los_Angeles", 33.94, -118.41)
    
    solutions = []
    
    # Create mock flight dates for demonstration
    from datetime import datetime
    
    # Mock FlightDate with minimal setup for departure time
    class MockFlightDate:
        def __init__(self, departure_hour):
            self.departure_datetime = datetime(2025, 12, 1, departure_hour, 0)
            # Mock schedule for properties
            self.schedule = type('obj', (object,), {
                'flight_code': f'XX{departure_hour}00',
                'route': type('obj', (object,), {'origin': jfk, 'destination': lax})()
            })()
    
    # Solution 1: Direct flight, expensive, morning departure
    sol1 = TravelSolution()
    sol1.total_price = 450.0
    sol1.total_travel_time = timedelta(hours=5, minutes=30)
    sol1.num_connections = 0
    sol1.available_seats = 10
    sol1.flights = [MockFlightDate(8)]  # 8 AM departure
    solutions.append(sol1)
    
    # Solution 2: One connection, cheaper, afternoon departure
    sol2 = TravelSolution()
    sol2.total_price = 320.0
    sol2.total_travel_time = timedelta(hours=8, minutes=15)
    sol2.num_connections = 1
    sol2.available_seats = 15
    sol2.flights = [MockFlightDate(14)]  # 2 PM departure
    solutions.append(sol2)
    
    # Solution 3: Two connections, cheapest, early morning
    sol3 = TravelSolution()
    sol3.total_price = 250.0
    sol3.total_travel_time = timedelta(hours=11, minutes=45)
    sol3.num_connections = 2
    sol3.available_seats = 20
    sol3.flights = [MockFlightDate(5)]  # 5 AM departure
    solutions.append(sol3)
    
    print("Available travel solutions:")
    for i, sol in enumerate(solutions, 1):
        print(f"  Option {i}: ${sol.total_price:.2f}, {sol.total_travel_time}, {sol.num_connections} connections")
    print()
    
    # Test 1: Simple cheapest choice
    print("TEST 1: Simple Choice (cheapest within WTP)")
    affordable = [s for s in solutions if s.total_price <= customer.willingness_to_pay]
    if affordable:
        chosen = min(affordable, key=lambda s: s.total_price)
        print(f"  Chosen: ${chosen.total_price:.2f} (cheapest)")
    print()
    
    # Test 2: MNL choice
    print("TEST 2: Multinomial Logit (MNL) Choice")
    mnl_model = MultinomialLogitModel()
    choice_set = ChoiceSet(own_solutions=solutions)
    
    # Calculate probabilities
    probs = mnl_model.calculate_choice_probabilities(choice_set, customer)
    print(f"  Choice probabilities:")
    for idx, prob in probs.items():
        if idx >= 0:
            print(f"    Option {idx+1}: {prob*100:.1f}%")
        else:
            print(f"    No purchase: {prob*100:.1f}%")
    
    # Simulate choice
    rng = np.random.default_rng(42)
    chosen = mnl_model.predict_choice(choice_set, customer, rng)
    if chosen:
        idx = solutions.index(chosen) + 1
        print(f"  Simulated choice: Option {idx} (${chosen.total_price:.2f})")
    else:
        print(f"  Simulated choice: No purchase")
    print()
    
    # Test 3: Buy-up/buy-down behavior
    print("TEST 3: Buy-Up/Buy-Down Behavior")
    buyupdown_model = BuyUpDownModel()
    
    preferred_price = 350.0
    higher_price = 450.0
    lower_price = 250.0
    
    # Test buy-up
    buy_up_decisions = []
    for _ in range(100):
        will_buy = buyupdown_model.will_buy_up(customer, preferred_price, higher_price, rng)
        buy_up_decisions.append(will_buy)
    
    buy_up_rate = sum(buy_up_decisions) / len(buy_up_decisions)
    print(f"  Buy-up probability (${preferred_price:.0f} → ${higher_price:.0f}): {buy_up_rate*100:.1f}%")
    
    # Test buy-down
    buy_down_decisions = []
    for _ in range(100):
        will_buy = buyupdown_model.will_buy_down(customer, preferred_price, lower_price, rng)
        buy_down_decisions.append(will_buy)
    
    buy_down_rate = sum(buy_down_decisions) / len(buy_down_decisions)
    print(f"  Buy-down probability (${preferred_price:.0f} → ${lower_price:.0f}): {buy_down_rate*100:.1f}%")
    print()
    
    # Test 4: Recapture
    print("TEST 4: Recapture Modeling")
    recapture_model = RecaptureModel()
    
    recapture_prob = recapture_model.get_recapture_probability(customer, has_loyalty=False)
    print(f"  Base recapture probability: {recapture_prob*100:.1f}%")
    
    recapture_prob_loyal = recapture_model.get_recapture_probability(customer, has_loyalty=True)
    print(f"  With loyalty: {recapture_prob_loyal*100:.1f}%")
    print()
    
    # Test 5: Enhanced choice model
    print("TEST 5: Enhanced Choice Model (MNL + Buy-Up/Down + Recapture)")
    enhanced_model = EnhancedChoiceModel()
    
    chosen = enhanced_model.predict_choice_with_behavior(
        choice_set, customer, BookingClass.M, rng
    )
    if chosen:
        idx = solutions.index(chosen) + 1
        print(f"  Enhanced choice: Option {idx} (${chosen.total_price:.2f})")
    else:
        print(f"  Enhanced choice: No purchase")
    print()


def demo_full_simulation():
    """Demonstrate full simulation with overbooking and choice models."""
    print("=" * 70)
    print("FULL SIMULATION WITH OVERBOOKING AND CHOICE")
    print("=" * 70)
    print()
    
    # Create simple network
    airports = [
        Airport("JFK", "New York JFK", "New York", "USA", "America/New_York", 40.64, -73.78),
        Airport("LAX", "Los Angeles", "Los Angeles", "USA", "America/Los_Angeles", 33.94, -118.41)
    ]
    
    route = Route(
        origin=airports[0],
        destination=airports[1],
        distance_km=3974
    )
    
    aircraft = Aircraft(
        type_code="738",
        name="Boeing 737-800",
        capacity={
            CabinClass.ECONOMY: 150
        }
    )
    
    schedule = FlightSchedule(
        airline_code="AA",
        flight_number="100",
        route=route,
        aircraft=aircraft,
        departure_time=time(8, 0),
        arrival_time=time(11, 30),
        days_of_week=[0, 1, 2, 3, 4, 5, 6],  # Daily
        valid_from=date.today(),
        valid_until=date.today() + timedelta(days=90)
    )
    
    # Create config with overbooking and MNL choice
    config = SimulationConfig(
        start_date=date.today(),
        end_date=date.today() + timedelta(days=30),
        overbooking_enabled=True,
        overbooking_method="critical_fractile",
        overbooking_risk_tolerance=0.05,
        choice_model="mnl",
        include_buyup_down=True,
        include_recapture=True,
        random_seed=42
    )
    
    print(f"Simulation configuration:")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Overbooking: {config.overbooking_enabled} ({config.overbooking_method})")
    print(f"  Choice model: {config.choice_model}")
    print(f"  Buy-up/down: {config.include_buyup_down}")
    print(f"  Recapture: {config.include_recapture}")
    print()
    
    print("Note: Full simulation would run here with:")
    print("  - Realistic demand generation")
    print("  - Overbooking limits calculated dynamically")
    print("  - Customer choice via MNL for each booking request")
    print("  - Show-up simulation at departure")
    print("  - Denied boarding handling")
    print()
    print("See examples/basic_example.py and competitive_simulation.py")
    print("for complete simulation examples.")
    print()


def main():
    """Run all demonstrations."""
    demo_overbooking()
    print("\n")
    demo_customer_choice()
    print("\n")
    demo_full_simulation()
    
    print("=" * 70)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  • Overbooking can increase revenue by 2-5% with proper risk management")
    print("  • MNL choice models capture realistic customer behavior")
    print("  • Buy-up/down behavior affects revenue by ±3%")
    print("  • Recapture modeling prevents overestimating spill")
    print()
    print("These features make the simulator match real airline operations.")


if __name__ == "__main__":
    main()
