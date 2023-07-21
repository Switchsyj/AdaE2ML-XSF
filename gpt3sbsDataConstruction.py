import os
import openai as ai
import random
import json
import jsonlines
import re, hashlib
ai.api_key = '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'
# API_KEY=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

com = re.compile(',')
# import datasets


def hash_func(text):
    """
    Import from "https://huggingface.co/datasets/codeparrot/github-code/blob/main/github_preprocessing.py"
    Args:
        text (_type_): _description_
    Returns:
        _type_: _description_
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def annotate(gen_text, slu_tags, selected_entities, selected_label):
    utterance = gen_text.split()
    for ent, lbl in zip(selected_entities, selected_label):
        ent = ent.lower()
        if ent in gen_text:
            try:
                st = utterance.index(ent.split()[0])
                ed = utterance.index(ent.split()[-1])
                slu_tags[st] = f'B-{lbl}'
                slu_tags[st+1: ed+1] = [f'I-{lbl}'] * (len(ent.split())-1)
            except:
                return None, None
    return utterance, slu_tags

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
                   "category": ["Electronics", "Clothing", "Ornament", "Home and Kitchen", "Health and Fitness", "Beauty and Personal Care", "Sports and Outdoors", "Books", "Toys and Games", "Jewelry and Accessories"],
                   "price": ["Nineteen dollars and ninety-nine cents", "Fifty dollars and fifty cents", "One hundred and twenty-nine dollars", "One hundred and fifty-five dollars", "Eighty-nine euros and fifty cents", "Between 50 and 100 dollars", "Under 10 dollars"],
                #    "country": ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France", "Japan", "Brazil", "India"],
                   "country": ["United States", "United States", "United Kingdom", "France", "Japan", "Australia", "Germany", "Italy", "Canada", "India"],
                   "city": ["New York City", "Los Angeles", "London", "Paris", "Tokyo", "Sydney", "Berlin", "Rome", "Toronto", "Mumbai"],
                   "state": ["New York", "California", "England", "Paris", "Tokyo", "New South Wales", "Berlin", "Rome", "Ontario", "Maharashtra"],
                   "color": ["Red", "Blue", "Green", "Yellow", "Black", "White", "Pink", "Purple", "Orange", "Brown"],
                   "material": ["Cotton", "Leather", "Silk", "Wool", "Denim", "Polyester", "Velvet", "Linen", "Nylon", "Satin"],
                   "timeRange": ["Today", "Tomorrow", "Next week", "This weekend", "In two days", "In a week", "Same-day delivery", "Within 1-3 business days", "Next-day delivery", "Express delivery in 24 hours"],
                   "payment_type": ["Credit card", "Debit card", "PayPal", "Apple Pay", "Google Pay", "Cash on delivery", "Gift card", "Cryptocurrency", "Bank transfer", "Mobile wallet"],
                   "comparing sort": ["Price", "Released date", "Popular items first", "Best-selling items first", "Customer ratings", "Relevance", "Alphabetical order", "Discount percentage", "Brand reputation"],
                   "inquiry_type": ["Order status", "Delivery concerns", "Technical support", "Account verification"],
                   }
    senario_2slot = {"searching": ['product_name', 'category', 'price', 'country', 'city', 'state', 'color', 'material'],
                     "ordering": ['product_name', 'timeRange', 'country', 'city', 'state', 'payment_type'],
                     "comparing": ['product_name', 'category', 'price', 'color', 'material', 'comparing sort'],
                     "payment": ['product_name', 'payment_type', 'price', 'timeRange'],
                     "customer service": ['inquiry_type', 'payment_type', 'product_name'],
                     "delivery": ['product_name', 'timeRange', 'country', 'city', 'state']}
                    #  "discount": ['product_name', 'price']}
    
    data_aug = []
    data_count = 0
    duplicate = 0
    hash_dict = dict()
    for i in range(800):
        # slot_label_choice = random.sample(generate_slot_list, 2) + random.sample(unseen_list, 2)
        scene = random.sample(["searching", "ordering", "comparing", "payment", "customer service", "delivery"], 1)[0]
        slot_label_choice = random.sample(senario_2slot[scene], min(len(senario_2slot[scene]), 2))
        print("============")
        print(slot_label_choice)
        print("============")
        print(f"scene: {scene}")
        print("============")
        selected_entities = []
        selected_label = []
        if any(s in ['country', 'state', 'city'] for s in slot_label_choice):
            random_int = random.sample(list(range(10)), 2)
            for i, _lbl in enumerate(slot_label_choice):
                if _lbl in ['country', 'state', 'city']:
                    selected_entities.extend([slot2entity[_lbl][k] for k in random_int])
                    selected_label.extend([_lbl] * 2)
                else:
                    selected_entities.extend(random.sample(slot2entity[_lbl], 2))
                    selected_label.extend([_lbl] * 2)
        else:
            for i, _lbl in enumerate(slot_label_choice):
                selected_entities.extend(random.sample(slot2entity[_lbl], 2))
                selected_label.extend([_lbl] * 2)
        print("============")
        print(f"entites: {selected_entities}")
        print("============")

        assert len(selected_entities) == len(selected_label)
        temp = list(zip(selected_entities, selected_label))
        random.shuffle(temp)
        selected_entities, selected_label = zip(*temp)
        usr_prompt = f"please give me 2 'imperative' utterances in '{scene}' e-commerce scenario containing exactly the following phrases: '{selected_entities[0]}', '{selected_entities[1]}'. The 2 utterances should in different from each other and itemized by '1. ', '2. ', etc. You should only return the itemized utterances."
        
        print(usr_prompt)
        gen_text = generate_gpt3_response(usr_prompt, print_output=True)
        print(gen_text)
        for item in gen_text.strip().split('\n'):
            if item.startswith('1.') or item.startswith('2.'):
                item = re.sub(com, '', item[2:].strip()).lower()
                item = item[:-1] if (item[-1] == '.' or item[-1] == '?' or item[-1] == '!') else item
                if not hash_dict.get(hash_func(item), False):
                    slu_tags = ['O'] * len(item.split())
                    utter, slu = annotate(item, slu_tags, selected_entities[0:2], selected_label[0:2])
                    hash_dict[hash_func(item)] = True
                else:
                    duplicate += 1
                    continue
                if utter is not None and slu is not None:
                    with jsonlines.open(f'data/constsf/ecommerce_{scene}.json', 'a') as f:
                        data_line = {"utterance": utter, "slu_tags": slu}
                        f.write(data_line)
                        data_count += 1
                
    print(f"======write {data_count} lines ======")