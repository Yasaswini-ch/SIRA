# 🎤 SIRA | 5-Minute Hackathon Demo Script

## ⏱️ 0:00 - 0:30 | THE HOOK (The Problem)
*   **(Action: Open the Landing Page `/`)**
*   **Speaker:** "Good morning/afternoon, we are Team GVPCEW. Every retail business, from local shops to supermarkets, loses significant revenue either due to **stockouts**—where customers can't find products—or **overstocking**, where capital is frozen in rotting inventory. We built **SIRA** (Smart Inventory Restock Advisor) to replace manual guesswork with predictive machine learning."

## ⏱️ 0:30 - 2:00 | SINGLE PREDICTION (The AI in Action)
*   **(Action: Navigate to `/dashboard`)**
*   **Speaker:** "Let's look at a live scenario. A store manager wants to check a **Premium Biscuit** line."
*   **Action:** Enter the following into the Predictor form:
    *   **Item Type:** Snack Foods
    *   **MRP:** 150
    *   **Outlet Type:** Supermarket Type1
    *   **Current Stock:** 20
*   **Action: Click 'Launch Prediction'**
*   **Speaker:** "Instantly, SIRA analyzes regional trends and historical demand. You see 3 outputs: Our **Demand Forecast** predicts 1,800 units this month; based on that, our Decision Engine calculates a **suggested restock quantity** of 450; and triggers a **Yellow Alert**, advising the manager to schedule a restock this week."

## ⏱️ 2:00 - 3:30 | BATCH UPLOADER (Scaling to the Business)
*   **(Action: Navigate to `/upload`)**
*   **Speaker:** "In a real warehouse, you don't check items one by one. You check thousands. With SIRA's **Batch Pipeline**, we can drop an entire store export CSV directly."
*   **Action: Drag and drop the test CSV. Click 'Analyze'.**
*   **Speaker:** "Note the **Red Alerts** at the top. SIRA automatically prioritizes the items with the highest deficit. By clicking 'Download Report', a procurement officer gets a filtered list of exactly what to buy *right now*."

## ⏱️ 3:30 - 4:30 | MODEL INSIGHTS (The Technical Edge)
*   **(Action: Scroll to the Charts in Dashboard)**
*   **Speaker:** "Our system isn't a black box. Our winning **Linear Regression** model achieves a clean RMSE of ~1132 and is optimized for generalization. As you see in the **Feature Importance** chart, Item Price (MRP) and Store Type are the strongest drivers. We compared this against Random Forest and XGBoost to ensure we avoid overfitting."

## ⏱️ 4:30 - 5:00 | CLOSING (The Commercial Vision)
*   **Speaker:** "SIRA isn't just a project; it's a replacement for manual Excel sheets. It offers **94% forecast precision** and a logic engine that manages risk during lead times. We are GVPCEW, and this is how we're making inventory management smart. Thank you."

---

## ❓ Q&A PREP: Likely Judge Questions

**Q: Why choose Linear Regression over XGBoost if XGBoost is 'smarter'?**
*   **A:** "In our testing, Linear Regression showed better generalization (lower CV RMSE). XGBoost tended to overfit on the specific noise of this dataset, so we prioritized reliability over training scores."

**Q: How do you handle lead times and safety stock?**
*   **A:** "We use a simplified EOQ (Economic Order Quantity) formula inside `utils.py` that calculates demand during the reorder period plus a dynamic safety buffer based on a 7-day lead time."

**Q: Is the data preprocessed for skewed distributions?**
*   **A:** "Yes, we handle missing weights via group-based medians and apply Standard Scaling to all numerical inputs (MRP, Visibility) to ensure equitable feature weighing."
