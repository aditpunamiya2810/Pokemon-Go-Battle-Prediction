import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Load the saved model
with open('pokemon_battle_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the Pokémon dataset
pokemon_data = pd.read_csv(r'C:\Users\Adit Punamiya\OneDrive\Desktop\Fight prediction\pokemons_data.csv', index_col=0)

# Function to normalize data and handle missing values
def preprocess_data(data_df):
    stats = ["Hit Points", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Legendary"]
    stats_df = pokemon_data[stats].T.to_dict("list")
    one = data_df.First_pokemon.map(stats_df)
    two = data_df.Second_pokemon.map(stats_df)
    temp_list = []
    for i in range(len(one)):
        temp_list.append(np.array(one[i]) - np.array(two[i]))
    new_test = pd.DataFrame(temp_list, columns=stats)
    
    # Impute missing values with mean
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    new_test_imputed = pd.DataFrame(imputer.fit_transform(new_test), columns=new_test.columns)
    return new_test_imputed

# Function to get Pokémon image from PokéAPI
def get_dream_world_sprite(pokemon_name):
    base_url = 'https://pokeapi.co/api/v2/pokemon/'
    response = requests.get(f'{base_url}{pokemon_name.lower()}')
    if response.status_code == 200:
        pokemon_data = response.json()
        dream_world_sprite_url = pokemon_data['sprites']['other']['dream_world']['front_default']
        return dream_world_sprite_url
    else:
        return None

# Function to display Pokémon logo
def display_pokemon_logo():
    st.image('580b57fcd9996e24bc43c52f.webp', width=700)
    st.write('')

# Streamlit app layout
def main():
    st.set_page_config(page_title='Pokémon Go Battle Predictor', page_icon='pokeball-logo-DC23868CA1-seeklogo.com.webp')
    display_pokemon_logo()
    st.title('Pokémon Go Battle Predictor')

    # User input for Pokémon names
    first_pokemon = st.text_input('Enter the name of the first Pokémon:')
    second_pokemon = st.text_input('Enter the name of the second Pokémon:')

    # Button to predict the winner
    if st.button('Predict Winner'):
        try:
            # Convert Pokémon names to IDs
            first_id = pokemon_data[pokemon_data['Name'] == first_pokemon].index.values[0]
            second_id = pokemon_data[pokemon_data['Name'] == second_pokemon].index.values[0]

            # Prepare input data for prediction
            input_data = pd.DataFrame([[first_id, second_id]], columns=['First_pokemon', 'Second_pokemon'])
            input_data_processed = preprocess_data(input_data)

            # Make prediction
            prediction = model.predict(input_data_processed)

            # Get the predicted winner's name
            winner_name = first_pokemon if prediction == 0 else second_pokemon

            # Display the predicted winner
            st.write(f'The predicted winner is: {winner_name}')

            # Fetch and display winner's Pokémon image
            winner_image_url = get_dream_world_sprite(winner_name)
            if winner_image_url:
                st.image(winner_image_url, caption=f'{winner_name} (Winner)', width=300)
            else:
                st.write('Error fetching Pokémon image.')

        except IndexError:
            st.write('Error: Pokémon not found in the dataset.')

if __name__ == '__main__':
    main()
