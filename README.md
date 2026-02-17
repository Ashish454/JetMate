# Flight-Booking-Chatbot

An AI-powered chatbot that streamlines flight bookings, answers FAQs, and holds small-talk using classic NLP techniques (TF–IDF vectorisation and cosine similarity). The bot matches user queries to the closest knowledge snippets and flight details contained in the provided CSV datasets.

---

## 📦 Repository Structure

```
.
├── HAI.py                   # Main application entry-point
├── Flight_Dataset.csv       # Sample flight information (used for lookup/matching)
├── QA_Dataset.csv           # FAQ/knowledge-base pairs for intent matching
├── SmallTalk_Dataset.csv    # Short conversational responses
└── README.md
```

---

## 🔧 Setup

1. Create and activate a virtual environment (recommended), then install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the three CSV files are in the same directory as `HAI.py` (filenames as listed above).

---

## ▶️ Run

```bash
python HAI.py
```

> The script loads the datasets, builds TF–IDF vectors, and responds to user inputs by retrieving the most similar entries via cosine similarity.

---

## 🗃️ Data

- **Flight_Dataset.csv** – Contains flight-related details used during booking/verification.
- **QA_Dataset.csv** – Question–answer pairs for booking-related FAQs.
- **SmallTalk_Dataset.csv** – Short, generic conversational replies.

> Column names/formats can be adapted inside `HAI.py` if needed.

---

## 📝 Notes

- This project relies on traditional IR/NLP methods (TF–IDF + cosine similarity) and does not require external model downloads.
- If you add new rows to the CSVs, simply re-run the script to refresh the vectors.

---

## 📄 Licence

Released under the MIT Licence.
