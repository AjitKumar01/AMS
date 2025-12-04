# Personalization in Airline Revenue Management

## Research Summary

Based on recent academic literature (e.g., Wittman & Belobaba, 2017; Wang et al., 2021), personalization in airline revenue management is shifting from static class-based pricing to dynamic offer generation.

### Key Concepts

1.  **Dynamic Offer Generation (DOG):** Instead of filing static fares, airlines generate offers in real-time based on the specific request context and customer attributes.
2.  **Ancillary Bundling:** Creating personalized bundles (e.g., "Business Plus" = Seat + Wifi + Lounge) increases conversion and total revenue per passenger.
3.  **Loyalty Integration:** Using loyalty status not just for rewards, but for pricing decisions (e.g., targeted discounts to retain high-value customers or prevent churn).
4.  **Willingness to Pay (WTP) Refinement:** Using customer data (past behavior, search context) to better estimate WTP than segment-level averages.

## Implementation in PyAirline RM

We have implemented a `PersonalizationEngine` that integrates into the core simulation loop.

### 1. Customer Enrichment
When a booking request is received, the engine enriches the `Customer` object with:
*   **Loyalty Tier:** (None, Silver, Gold, Platinum) assigned probabilistically based on the customer segment.
*   **Ancillary Preferences:** (Bags, Wifi, Lounge, etc.) assigned based on segment behavior.

### 2. Personalized Offer Generation
The engine intercepts the standard travel solutions and applies:
*   **Loyalty Discounts:** High-tier members (Gold/Platinum) may receive automatic discounts on standard fares.
*   **Dynamic Bundling:** If a customer has specific ancillary preferences, the engine creates a "Bundle Offer" that includes these items at a discounted package rate compared to buying them separately.

### 3. Simulation Integration
*   **Input:** The user can enable "Personalized Offers" in the UI.
*   **Process:** The simulator passes every request through the `PersonalizationEngine`.
*   **Output:** The `ChoiceModel` evaluates these new personalized offers (bundles/discounts) alongside standard offers.

## Future Work
*   **Continuous Pricing:** Implement fully continuous pricing where the fare is a function of $f(WTP, Capacity)$ rather than discrete buckets.
*   **ML-based Personalization:** Use the `NeuralNetwork` forecaster to predict individual conversion probabilities for specific bundles.
