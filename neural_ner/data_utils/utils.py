def create_freq_map(item_list):
    assert type(item_list) is list
    freq_map = {}
    for items in item_list:
        for item in items:
            if item not in freq_map:
                freq_map[item] = 1
            else:
                freq_map[item] += 1
    return freq_map

def create_mapping(freq_map):
    sorted_items = sorted(freq_map.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item