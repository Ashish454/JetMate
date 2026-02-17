# Flight Booking Chatbot which also handles small talk and QnA

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime

# Load and combine the datasets
def load_and_combine_datasets(small_talk, qna, flight_data):
    try:
        dataset1 = pd.read_csv(small_talk, encoding='ISO-8859-1')
        dataset2 = pd.read_csv(qna, encoding='ISO-8859-1')
        flight_dataset = pd.read_csv(flight_data, encoding='ISO-8859-1')
        combined_dataset = pd.concat([dataset1, dataset2], ignore_index=True)

        # Ensure 'Question' and 'Answer' columns exist in QA and Small Talk datasets
        if 'Question' not in combined_dataset.columns or 'Answer' not in combined_dataset.columns:
            raise ValueError("Both datasets must contain 'Question' and 'Answer' columns.")

        # Fill missing values if any
        combined_dataset = combined_dataset.fillna("Unknown")

        # Preprocess questions to be lowercase
        combined_dataset['Question'] = combined_dataset['Question'].str.lower()

        print("Datasets loaded and combined successfully.")
        return combined_dataset, flight_dataset
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None

# Preprocess and Vectorize the Combined Data
def preprocess_and_vectorize(dataset):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset['Question'])
    return vectorizer, tfidf_matrix


# Get the Best Response
def get_best_response(user_input, dataset, vectorizer, tfidf_matrix, user_name, flight_dataset):
    user_input = user_input.strip().lower()

    # Check if the user is asking for their name
    if user_input == "what is my name":
        return f"Your name is {user_name}." if user_name else "I don't know your name yet."

    # Flight booking functionality trigger
    if 'book a flight' in user_input or 'flight' in user_input:
        return flight_booking(user_input, flight_dataset, user_name)

    # Otherwise, process as usual
    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()

    if similarity_scores[0][best_match_idx] > 0.2:  # Set a threshold for response relevance
        response = dataset.iloc[best_match_idx]['Answer']
        return response.replace("{name}", user_name)  # Insert the user's name where relevant
    else:
        return f"I’m sorry, {user_name}, I didn’t quite understand that. Can you rephrase?"

# Handle the flight booking process
def flight_booking(user_input, flight_dataset, user_name):
    # Step 1: Ask for origin city
    while True:
        print("Chatbot: Please specify your origin city")
        origin = input(f"{user_name}: ").strip().title()
        if origin in flight_dataset['Origin'].unique():
            break
        else:
            print(f"Chatbot: Sorry, we do not operate flights from {origin}. Please try another city.")

    # Step 2: Ask for destination city
    while True:
        print("Chatbot: Please specify your destination city")
        destination = input(f"{user_name}: ").strip().title()
        if destination in flight_dataset['Destination'].unique():
            break
        else:
            print(f"Chatbot: Sorry, we do not operate flights to {destination}. Please try another city.")

    # Step 3: Ask if it's a single or return ticket
    while True:
        print("Chatbot: Is this a single or return ticket? (Please type 'single' or 'return')")
        ticket_type = input(f"{user_name}: ").strip().lower()
        if ticket_type in ["single", "return"]:
            break
        else:
            print("Chatbot: Sorry, I didn't understand that. Please specify if it's a single or return ticket.")

    # Step 4: Ask for dates
    while True:
        if ticket_type == "single":
            print("Chatbot: Please provide your departure date in DD/MM/YYYY format.")
            dates_input = input(f"{user_name}: ").strip()
            try:
                departure_date = datetime.strptime(dates_input, '%d/%m/%Y')
                return_date = None
                break
            except ValueError:
                print("Chatbot: Sorry, the date format seems incorrect. Please try again.")
        else:  # Return ticket
            print(
                "Chatbot: Please provide your departure and return dates in the format shown (e.g., DD/MM/YYYY, DD/MM/YYYY).")
            dates_input = input(f"{user_name}: ").strip()
            try:
                departure_date_str, return_date_str = dates_input.split(',')
                departure_date = datetime.strptime(departure_date_str.strip(), '%d/%m/%Y')
                return_date = datetime.strptime(return_date_str.strip(), '%d/%m/%Y')
                if return_date > departure_date:
                    break
                else:
                    print("Chatbot: The return date must be after the departure date. Please try again.")
            except (ValueError, IndexError):
                print(
                    "Chatbot: Sorry, the date format seems incorrect. Please try again with both dates separated by a comma.")

    # Step 5: Check availability for departure and return flights
    available_departure_flights = flight_dataset[
        (flight_dataset['Origin'] == origin) &
        (flight_dataset['Destination'] == destination) &
        (flight_dataset['Departure Date'] == departure_date.strftime('%d/%m/%Y'))
        ]

    available_return_flights = pd.DataFrame()
    if ticket_type == "return" and return_date:
        available_return_flights = flight_dataset[
            (flight_dataset['Origin'] == destination) &
            (flight_dataset['Destination'] == origin) &
            (flight_dataset['Departure Date'] == return_date.strftime('%d/%m/%Y'))
            ]

    # Handle availability for single tickets
    if ticket_type == "single":
        if not available_departure_flights.empty:
            flight_id = available_departure_flights.iloc[0]['Flight ID']
            print(
                f"Chatbot: Great! We have a flight available from {origin} to {destination} on {departure_date.strftime('%d/%m/%Y')}. Would you like to confirm this booking? (yes/no)")
            confirm = input(f"{user_name}: ").strip().lower()
            if confirm == "yes":
                return f"Your booking from {origin} to {destination} on {departure_date.strftime('%d/%m/%Y')} (Flight ID: {flight_id}) has been confirmed! Anything else I can assist you with?"
            else:
                return "Alright, let me know if there's anything else I can assist you with."
        else:
            print(
                f"Chatbot: Unfortunately, no flights are available on {departure_date.strftime('%d/%m/%Y')}. Are you flexible with your dates? (yes/no)")
            flexible = input(f"{user_name}: ").strip().lower()
            if flexible == "yes":
                alternate_departure_flights = flight_dataset[
                    (flight_dataset['Origin'] == origin) &
                    (flight_dataset['Destination'] == destination) &
                    (flight_dataset['Departure Date'] != departure_date.strftime('%d/%m/%Y'))
                    ]
                if not alternate_departure_flights.empty:
                    new_departure_date = alternate_departure_flights['Departure Date'].iloc[0]
                    flight_id = alternate_departure_flights.iloc[0]['Flight ID']
                    print(
                        f"Chatbot: We have a flight available on {new_departure_date}. Would you like to confirm this booking? (yes/no)")
                    confirm = input(f"{user_name}: ").strip().lower()
                    if confirm == "yes":
                        return f"Your booking from {origin} to {destination} on {new_departure_date} (Flight ID: {flight_id}) has been confirmed! Anything else I can assist you with?"
                    else:
                        return "Alright, let me know if there's anything else I can assist you with."
                else:
                    return "Sorry, there are no alternative flights available. Would you like help with something else?"
            else:
                return "Alright, let me know if there's anything else I can assist you with."

    # Handle availability for return tickets
     # If flight is available on both the dates mentioned by the user
    if ticket_type == "return":
        if not available_departure_flights.empty and not available_return_flights.empty:
            departure_flight_id = available_departure_flights.iloc[0]['Flight ID']
            return_flight_id = available_return_flights.iloc[0]['Flight ID']
            print(
                f"Chatbot: Great! We have flights available for both departure and return. Departure on {departure_date.strftime('%d/%m/%Y')} and return on {return_date.strftime('%d/%m/%Y')}. Would you like to confirm this booking? (yes/no)")
            confirm = input(f"{user_name}: ").strip().lower()
            if confirm == "yes":
                return f"Your booking from {origin} to {destination} on {departure_date.strftime('%d/%m/%Y')} (Flight ID: {departure_flight_id}) and return on {return_date.strftime('%d/%m/%Y')} (Flight ID: {return_flight_id}) has been confirmed! Anything else I can assist you with?"
            else:
                return "Alright, let me know if there's anything else I can assist you with."

        # If flight is only available on the departure date mentioned by the user
        elif not available_departure_flights.empty:
            departure_flight_id = available_departure_flights.iloc[0]['Flight ID']
            print(
                f"Chatbot: There is a flight for departure on {departure_date.strftime('%d/%m/%Y')} but no flight for return. Are you flexible with your return date? (yes/no)")
            flexible = input(f"{user_name}: ").strip().lower()
            if flexible == "yes":
                alternate_return_flights = flight_dataset[
                    (flight_dataset['Origin'] == destination) &
                    (flight_dataset['Destination'] == origin) &
                    (flight_dataset['Departure Date'] != return_date.strftime('%d/%m/%Y'))
                    ]
                if not alternate_return_flights.empty:
                    new_return_date = alternate_return_flights['Departure Date'].iloc[0]
                    return_flight_id = alternate_return_flights.iloc[0]['Flight ID']
                    # Check if the new return date is after the departure date
                    if datetime.strptime(new_return_date, '%d/%m/%Y') > departure_date:
                        print(
                            f"Chatbot: We have a return flight available on {new_return_date}. Would you like to confirm this booking? (yes/no)")
                        confirm = input(f"{user_name}: ").strip().lower()
                        if confirm == "yes":
                            return f"Your booking from {origin} to {destination} on {departure_date.strftime('%d/%m/%Y')} (Flight ID: {departure_flight_id}) and return on {new_return_date} (Flight ID: {return_flight_id}) has been confirmed! Anything else I can assist you with?"
                        else:
                            return "Alright, let me know if there's anything else I can assist you with."
                    else:
                        return f"The adjusted return date ({new_return_date}) is before the departure date. No valid return flights available. Would you like help with something else?"
                else:
                    return "Sorry, there are no alternative return flights available. Would you like help with something else?"
            else:
                return "Alright, let me know if there's anything else I can assist you with."

        # If flight is only available on the return date mentioned by the user
        elif not available_return_flights.empty:
            return_flight_id = available_return_flights.iloc[0]['Flight ID']
            print(
                f"Chatbot: There is a flight for return on {return_date.strftime('%d/%m/%Y')} but no flight for departure. Are you flexible with your departure date? (yes/no)")
            flexible = input(f"{user_name}: ").strip().lower()
            if flexible == "yes":
                alternate_departure_flights = flight_dataset[
                    (flight_dataset['Origin'] == origin) &
                    (flight_dataset['Destination'] == destination) &
                    (flight_dataset['Departure Date'] != departure_date.strftime('%d/%m/%Y'))
                    ]
                if not alternate_departure_flights.empty:
                    new_departure_date = alternate_departure_flights['Departure Date'].iloc[0]
                    departure_flight_id = alternate_departure_flights.iloc[0]['Flight ID']
                    # Check if the new departure date is before the return date
                    if datetime.strptime(new_departure_date, '%d/%m/%Y') < return_date:
                        print(
                            f"Chatbot: We have a departure flight available on {new_departure_date}. Would you like to confirm this booking? (yes/no)")
                        confirm = input(f"{user_name}: ").strip().lower()
                        if confirm == "yes":
                            return f"Your booking from {origin} to {destination} on {new_departure_date} (Flight ID: {departure_flight_id}) and return on {return_date.strftime('%d/%m/%Y')} (Flight ID: {return_flight_id}) has been confirmed! Anything else I can assist you with?"
                        else:
                            return "Alright, let me know if there's anything else I can assist you with."
                    else:
                        return f"The adjusted departure date ({new_departure_date}) is after the return date. No valid departure flights available. Would you like help with something else?"
                else:
                    return "Sorry, there are no alternative departure flights available. Would you like help with something else?"
            else:
                return "Alright, let me know if there's anything else I can assist you with."

        # There on no flight matching with both the dates mentioned by the user
        else:
            print(
                f"Chatbot: Unfortunately, no flights are available for either departure or return. Are you flexible with your dates? (yes/no)")
            flexible = input(f"{user_name}: ").strip().lower()
            if flexible == "yes":
                alternate_departure_flights = flight_dataset[
                    (flight_dataset['Origin'] == origin) &
                    (flight_dataset['Destination'] == destination) &
                    (flight_dataset['Departure Date'] != departure_date.strftime('%d/%m/%Y'))
                    ]
                alternate_return_flights = flight_dataset[
                    (flight_dataset['Origin'] == destination) &
                    (flight_dataset['Destination'] == origin) &
                    (flight_dataset['Departure Date'] != return_date.strftime('%d/%m/%Y'))
                    ]
                if not alternate_departure_flights.empty and not alternate_return_flights.empty:
                    new_departure_date = alternate_departure_flights['Departure Date'].iloc[0]
                    new_return_date = alternate_return_flights['Departure Date'].iloc[0]
                    departure_flight_id = alternate_departure_flights.iloc[0]['Flight ID']
                    return_flight_id = alternate_return_flights.iloc[0]['Flight ID']
                    # Validate the adjusted dates
                    if datetime.strptime(new_return_date, '%d/%m/%Y') > datetime.strptime(new_departure_date,
                                                                                          '%d/%m/%Y'):
                        print(
                            f"Chatbot: We have flights available. Departure on {new_departure_date} and return on {new_return_date}. Would you like to confirm this booking? (yes/no)")
                        confirm = input(f"{user_name}: ").strip().lower()
                        if confirm == "yes":
                            return f"Your booking from {origin} to {destination} on {new_departure_date} (Flight ID: {departure_flight_id}) and return on {new_return_date} (Flight ID: {return_flight_id}) has been confirmed! Anything else I can assist you with?"
                        else:
                            return "Alright, let me know if there's anything else I can assist you with."
                    else:
                        return f"The adjusted return date ({new_return_date}) is before the departure date. No valid flights available. Would you like help with something else?"
                else:
                    return "Sorry, there are no alternative flights available. Would you like help with something else?"
            else:
                return "Alright, let me know if there's anything else I can assist you with."


# Main Chatbot Function
def chatbot(small_talk, qna, flight_data):
    combined_dataset, flight_dataset = load_and_combine_datasets(small_talk, qna, flight_data)
    if combined_dataset is None or flight_dataset is None:
        return

    vectorizer, tfidf_matrix = preprocess_and_vectorize(combined_dataset)
    print("Chatbot: Hi there! What's your name?")

    user_name = ""
    while not user_name:  # Keep asking until a valid name is provided
        input_name = input("You: ").strip().title()

        # Extract name from sentences like "Hi! My name is Aditi"
        if "my name is" in input_name.lower():
            user_name = input_name.lower().split("my name is")[-1].strip().title()
        elif len(input_name.split()) == 1:  # If the input is a single word, treat it as a name
            user_name = input_name
        else:
            print("Chatbot: Please provide me with your actual name, not a greeting or unrelated statement.")
    print(f"Chatbot: Nice to meet you, {user_name}! How can I help you today?")
    while True:
        user_input = input(f"{user_name}: ")
        if user_input.strip().lower() in ["exit", "quit", "bye"]:
            print(f"Chatbot: Goodbye, {user_name}! Have a great day!")
            break
        response = get_best_response(user_input, combined_dataset, vectorizer, tfidf_matrix, user_name, flight_dataset)
        print(f"Chatbot: {response}")

# Paths to the datasets
small_talk = 'SmallTalk_Dataset.csv'
qna = 'QA_Dataset.csv'
flight_data = 'Flight_Dataset.csv'

# Run the chatbot
chatbot(small_talk, qna, flight_data)