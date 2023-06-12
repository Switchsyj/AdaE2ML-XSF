import os
import openai as ai
import random
import json
import jsonlines
ai.api_key = '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'
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
        temperature=0.6,            # Level of creativity in the response
        prompt=user_text,           # What the user typed in
        max_tokens=1024,             # Maximum tokens in the prompt AND response
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
                     "delivery": ['product_name', 'timeRange', 'country', 'city', 'state']}
                    #  "discount": ['product_name', 'price']}
    
    data_aug = []
    for i in range(100):
        # slot_label_choice = random.sample(generate_slot_list, 2) + random.sample(unseen_list, 2)
        scene = random.sample(["searching", "ordering", "comparing", "payment & refund & reorder", "customer service", "delivery"], 1)[0]
        slot_label_choice = random.sample(senario_2slot[scene], min(len(senario_2slot[scene]), 4))
        print("============")
        print(slot_label_choice)
        print("============")
        print(f"scene: {scene}")
        print("============")
        # usr_prompt = f"I want to generate one slot-filling data 'from e-commerce website' under '{scene}' scene. you can choose one or several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}, {slot_label_choice[2]}, {slot_label_choice[3]}'. You should return natural language utterance and slots in 'json' format and garantee it is true."
        if len(senario_2slot[scene]) < 4:
            usr_prompt = f"Give me an examples with `imperative` instructions. Suppose you are talking to an e-commercial assistant asking for '{scene}'. You can choose several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}'. You should generate instructive natural language utterance and slots using the chosen labels. You should return 'json' format data and garantee it is true under this dialogue conversation."
        else:
            usr_prompt = f"Give me an examples with `imperative` instructions. Suppose you are talking to an e-commercial assistant asking for '{scene}'. You can choose several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}, {slot_label_choice[2]}, {slot_label_choice[3]}'. You should generate instructive natural language utterance and slots using the chosen labels. You should return 'json' format data and garantee it is true under this dialogue conversation."
        usr_prompt += '{"scene": "delivery", "utterance": "Please deliver my order to city Los Angeles in state California, country U.S.A .", "slots": [{"slot_name": "city", "slot_value": "Los Angeles"}, {"slot_name": "state", "slot_value": "California"}, {"slot_name": "country", "slot_value": "U.S.A"}]}, {"scene": "searching", {"scene": "payment & refund & reorder", "utterance": "Pay for a 256GB black iphone-14 using my Visa card.", "slots": [{"slot_name": "product_name", "slot_value": "iphone-14"}, {"slot_name": "material", "slot_value": "black"}, {"slot_name": "size", "slot_value": "256GB"}, {"slot_name": "payment_type", "slot_value": "Visa card"}]}, {"scene": "refund & reorder", "utterance": "Refund my order with order number 7890 and reorder the same product.", "slots": [{"slot_name": "order_number", "slot_value": "7890"}, {"slot_name": "inquiry_type", "slot_value": "refund & reorder"}]}\n generate data: '

        print(usr_prompt)
        gen_text = generate_gpt3_response(usr_prompt, print_output=True)
        data_aug.append(gen_text)
    data_count = 0
    with jsonlines.open('data/constsf/ecommerce_0002.json', 'a') as f:
        for data in data_aug:
            try:
                f.write(json.loads(data))
                data_count += 1
            except:
                print(data)
    print(f"======write {data_count} lines ======")