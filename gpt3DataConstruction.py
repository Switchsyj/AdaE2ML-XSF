import os
import openai as ai
import random
import json
import jsonlines
# ai.api_key = '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'
# API_KEY=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

domain2slot = {
    "AddToPlaylist": ['[PAD]', 'music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['[PAD]', 'city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['[PAD]', 'city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['[PAD]', 'genre', 'music_item', 'service', 'year', 'playlist', 'album', 'sort', 'track', 'artist'],
    "RateBook": ['[PAD]', 'object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['[PAD]', 'object_name', 'object_type'],
    "SearchScreeningEvent": ['[PAD]', 'timeRange', 'movie_type', 'object_location_type', 'object_type', 'location_name', 'spatial_relation', 'movie_name']
}

slot_list = ['artist', \
            'city', 'timeRange', 'country', 'state', \
            'city', 'state', 'timeRange', 'country', \
            'artist', \
            'timeRange']

def generate_gpt3_response(user_text, print_output=False):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    completions = ai.Completion.create(
        engine='text-davinci-003',  # Determines the quality, speed, and cost.
        temperature=0.8,            # Level of creativity in the response
        prompt=user_text,           # What the user typed in
        max_tokens=512,             # Maximum tokens in the prompt AND response
        n=1,                        # The number of completions to generate
        stop=None,                  # An optional setting to control response generation
    )

    # Displaying the output can be helpful if things go wrong
    if print_output:
        print(completions)

    # Return the first choice's text
    return completions.choices[0].text

if __name__ == '__main__':
    # unique slot list
    generate_slot_list = list(set(slot_list))
    # seen_list = ['sort', 'city', 'timeRange', 'country', 'state']
    unseen_list = ['product_name', 'color', 'payment_type', 'inquiry_type', 'category', 'price', 'material', 'order_number']
    senario_2slot = {"searching": ['product_name', 'category', 'price', 'country', 'city', 'state', 'color', 'material'],
                     "ordering": ['product_name', 'timeRange', 'country', 'city', 'state', 'payment_type'],
                     "comparing": ['product_name', 'category', 'price', 'color', 'material', 'sort'],
                     "payment & refund & reorder": ['inquiry_type', 'product_name', 'payment_type', 'price', 'timeRange', 'order_number'],
                     "customer service": ['inquiry_type', 'payment_type', 'product_name', 'order_number'],
                     "delivery": ['product_name', 'timeRange', 'country', 'city', 'state'],
                     "discount": ['product_name', 'price']}
    
    data_aug = []
    for i in range(100):
        # slot_label_choice = random.sample(generate_slot_list, 2) + random.sample(unseen_list, 2)
        scene = random.sample(["searching", "ordering", "comparing", "payment & refund & reorder", "customer service", "delivery", "discount"], 1)[0]
        slot_label_choice = random.sample(senario_2slot[scene], min(len(senario_2slot[scene]), 4))
        print("============")
        print(slot_label_choice)
        print("============")
        print(f"scene: {scene}")
        print("============")
        # usr_prompt = f"I want to generate one slot-filling data 'from e-commerce website' under '{scene}' scene. you can choose one or several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}, {slot_label_choice[2]}, {slot_label_choice[3]}'. You should return natural language utterance and slots in 'json' format and garantee it is true."
        if len(senario_2slot[scene]) < 4:
            usr_prompt = f"Suppose you are talking to an e-commercial assistant asking for '{scene}'. you can choose one or several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}'. You should return natural language utterance and slots in 'json' format and garantee it is true under this dialogue conversation."
        else:
            usr_prompt = f"Suppose you are talking to an e-commercial assistant asking for '{scene}'. you can choose one or several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}, {slot_label_choice[2]}, {slot_label_choice[3]}'. You should return natural language utterance and slots in 'json' format and garantee it is true under this dialogue conversation."
        usr_prompt += 'Example: {"scene": "payment & refund & reorder", "utterance": "Please help me order a 128GB red iphone-14 on Saturday.", "slots": [{"slot_name": "time_range", "slot_value": "Saturday"}, {"slot_name": "product_name", "slot_value": "iphone-14"}]}, {"scene": "searching", "utterance": "I want to buy a laptop, please help me search with price lower than 99 dollars.", "slots": [{"slot_name": "price", "slot_value": "99 dollars"}, {"slot_name": "category", "slot_value": "laptop"}]}, {"scene": "comparing", "utterance": "Compare two laptops Dell XPS 13 and Macbook Air in price and performance ratio.", "slots": [{"slot_name": "product_name", "slot_value": "Dell XPS 13"}, {"slot_name": "product_name", "slot_value": "Macbook Air"}, {"slot_name": "category", "slot_value": "laptop"}, {"slot_name": "sort", "slot_value": "price and performance ratio"}]}'
        print(usr_prompt)
        # gen_text = generate_gpt3_response(usr_prompt, print_output=True)
        # data_aug.append(gen_text)
    # data_count = 0
    # with jsonlines.open('data/constsf/ecommerce.json', 'w') as f:
    #     for data in data_aug:
    #         try:
    #             f.write(json.loads(data))
    #             data_count += 1
    #         except:
    #             print(data)
    # print(f"======write {data_count} lines ======")