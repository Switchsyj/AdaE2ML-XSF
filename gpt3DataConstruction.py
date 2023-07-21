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
    
    slot2entity = {"product_name": ["iPhone 12 Pro", "MacBook Pro", "Samsung Galaxy S21", "Nike Air Max", "Sony PlayStation 5", "Gucci handbag", "Canon EOS R5", "Xbox Series X", "Adidas Ultraboost", "Dyson V11 vacuum cleaner"],
                   "category": ["Electronics", "Clothing", "ornament", "Home and Kitchen"],
                   "price": ["Nineteen dollars and ninety-nine cents", "Fifty dollars and fifty cents", "One hundred and twenty-nine dollars", "One hundred and fifty-five dollars"],
                #    "country": ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France", "Japan", "Brazil", "India"],
                   "country": ["United States", "United States", "United Kingdom", "France", "Japan", "Australia", "Germany", "Italy", "Canada", "India"],
                   "city": ["New York City", "Los Angeles", "London", "Paris", "Tokyo", "Sydney", "Berlin", "Rome", "Toronto", "Mumbai"],
                   "state": ["New York", "California", "England", "Paris", "Tokyo", "New South Wales", "Berlin", "Rome", "Ontario", "Maharashtra"],
                   "color": ["Red", "Blue", "Green", "Yellow", "Black", "White", "Pink", "Purple", "Orange", "Brown"],
                   "material": ["Cotton", "Leather", "Silk", "Wool", "Denim", "Polyester", "Velvet", "Linen", "Nylon", "Satin"],
                   "timeRange": ["Today", "Tomorrow", "Next week", "This weekend", "In two days", "In a week", "Same-day delivery", "Within 1-3 business days", "Next-day delivery", "Express delivery in 24 hours"],
                   "payment_type": ["Credit card", "Debit card", "PayPal", "Apple Pay", "Google Pay", "Cash on delivery", "Gift card", "Cryptocurrency", "Bank transfer", "Mobile wallet"],
                   "comparing sort": ["Price low to high", "Price high to low", "Newest first", "Popular items first", "Best-selling items first", "Customer ratings", "Relevance", "Alphabetical order", "Discount percentage", "Brand reputation"],
                   "inquiry_type": ["Order status", "Delivery concerns", "Technical support", "Account verification"],
                   }
    senario_2slot = {"searching": ['product_name', 'category', 'price', 'country', 'city', 'state', 'color', 'material'],
                     "ordering": ['product_name', 'timeRange', 'country', 'city', 'state', 'payment_type'],
                     "comparing": ['product_name', 'category', 'price', 'color', 'material', 'comparing sort'],
                     "payment & refund & reorder": ['product_name', 'payment_type', 'price', 'timeRange'],
                     "customer service": ['inquiry_type', 'payment_type', 'product_name'],
                     "delivery": ['product_name', 'timeRange', 'country', 'city', 'state']}
                    #  "discount": ['product_name', 'price']}
    
    data_aug = []
    data_count = 0
    for i in range(1500):
        # slot_label_choice = random.sample(generate_slot_list, 2) + random.sample(unseen_list, 2)
        scene = random.sample(["searching", "ordering", "comparing", "payment & refund & reorder", "customer service", "delivery"], 1)[0]
        slot_label_choice = random.sample(senario_2slot[scene], min(len(senario_2slot[scene]), 2))
        print("============")
        print(slot_label_choice)
        print("============")
        print(f"scene: {scene}")
        print("============")
        # usr_prompt = f"I want to generate one slot-filling data 'from e-commerce website' under '{scene}' scene. you can choose one or several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}, {slot_label_choice[2]}, {slot_label_choice[3]}'. You should return natural language utterance and slots in 'json' format and garantee it is true."
        # if len(senario_2slot[scene]) < 4:
        #     usr_prompt = f"Give me an examples with `imperative` instructions. Suppose you are talking to an e-commercial assistant asking for '{scene}'. You can choose several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}'. You should generate instructive natural language utterance and slots using the chosen labels. You should return 'json' format data and garantee it is true under this dialogue conversation."
        # else:
        #     usr_prompt = f"Give me an examples with `imperative` instructions. Suppose you are talking to an e-commercial assistant asking for '{scene}'. You can choose several labels from below: '{slot_label_choice[0]}, {slot_label_choice[1]}'. You should generate instructive natural language utterance and slots using the chosen labels. You should return 'json' format data and garantee it is true under this dialogue conversation."
        usr_prompt = f"Suppose you are a data annonator, please generate `one` slot filling data in `{scene}` domain, with several labels from below: `{slot_label_choice[0]}, {slot_label_choice[1]}`. You should generate `imperative instructive` natural language utterance `without subject` and slots using the chosen labels. You should return `json` format data and garantee it is true under this dialogue conversation."
        # usr_prompt += '{"scene": "delivery", "utterance": "Please deliver my order to city Los Angeles in state California, country U.S.A .", "slots": [{"slot_name": "city", "slot_value": "Los Angeles"}, {"slot_name": "state", "slot_value": "California"}, {"slot_name": "country", "slot_value": "U.S.A"}]}, {"scene": "searching", {"scene": "payment & refund & reorder", "utterance": "Pay for a 256GB black iphone-14 using my Visa card.", "slots": [{"slot_name": "product_name", "slot_value": "iphone-14"}, {"slot_name": "material", "slot_value": "black"}, {"slot_name": "size", "slot_value": "256GB"}, {"slot_name": "payment_type", "slot_value": "Visa card"}]}, {"scene": "refund & reorder", "utterance": "Refund my order with order number 7890 and reorder the same product.", "slots": [{"slot_name": "order_number", "slot_value": "7890"}, {"slot_name": "inquiry_type", "slot_value": "refund & reorder"}]}\n generate data: '
        # usr_prompt += '{"scene": "delivery", "utterance": "Please deliver my order to city Los Angeles in state California, country U.S.A.", "slots": [{"slot_name": "city", "slot_value": "Los Angeles"}, {"slot_name": "state", "slot_value": "California"}, {"slot_name": "country", "slot_value": "U.S.A"}]}, {"scene": "searching", {"scene": "payment & refund & reorder", "utterance": "Pay for a 256GB black iphone-14 using my Visa card.", "slots": [{"slot_name": "product_name", "slot_value": "iphone-14"}, {"slot_name": "material", "slot_value": "black"}, {"slot_name": "size", "slot_value": "256GB"}, {"slot_name": "payment_type", "slot_value": "Visa card"}]}, {"scene": "searching", "utterance": "Search for clothes from California, U.S.A, blue, and cotton.", "slots": [{"slot_name": "state", "slot_value": "California"}, {"slot_name": "country", "slot_value": "U.S.A"}, {"slot_name": "color", "slot_value": "blue"}, {"slot_name": "material", "slot_value": "cotton"}]}]\n generate data: '
        # {"scene": "comparing", "utterance": "Compare the color, product_name, sort, material of the products.", "slots": [{"slot_name": "color", "slot_value": ""}, {"slot_name": "product_name", "slot_value": ""}, {"slot_name": "sort", "slot_value": ""}, {"slot_name": "material", "slot_value": ""}]}
        usr_prompt += 'Example: {"scene": "delivery", "utterance": "Deliver my order to Los Angeles in California, U.S.A .", "slots": [{"slot_name": "city", "slot_value": "Los Angeles"}, {"slot_name": "state", "slot_value": "California"}, {"slot_name": "country", "slot_value": "U.S.A"}]}, {"scene": "searching", "utterance": "Search for a product with wood in the furniture, with a range of $100 to $200.", "slots": [{"slot_name": "product_name", "slot_value": ""}, {"slot_name": "material", "slot_value": "wood"}, {"slot_name": "price", "slot_value": "$100-$200"}, {"slot_name": "category", "slot_value": "furniture"}]}\n `Do not repeat` the above examples, generate data: '

        print(usr_prompt)
        gen_text = generate_gpt3_response(usr_prompt, print_output=True)
        data_aug.append(gen_text)
        
        with jsonlines.open('data/constsf/ecommerce_0008.json', 'a') as f:
            gen_text = gen_text.strip()
            if gen_text[-1] in [',', '.', '?', '!']:
                gen_text = gen_text[:-1]
            try:
                f.write(json.loads(gen_text))
                data_count += 1
            except:
                print(f"load data failed:\n {gen_text}")
                
    print(f"======write {data_count} lines ======")