# This script contains functions for preprocessing order data
import pandas as pd
import numpy as np
import re


# fill in the NaNs resulting from  a per-order df style.
def desparsify_data_in_place(order_dataframe):
    order_dataframe.lineitem_sku = order_dataframe.lineitem_sku.astype(
        str).str.replace('nan', 'None')
    order_dataframe[['lineitem_price', 'lineitem_compare_at_price']] = \
        order_dataframe[['lineitem_price', 'lineitem_compare_at_price']].fillna(
            method='ffill', axis=1)
    order_dataframe.fillna(method='ffill', inplace=True)

# The second function should take in the dataframe and a list of column names
# that ought to be datetimes.
# Note: should change those columns to datetime IN PLACE


def convert_datetimes_in_place(order_dataframe, datetime_list):
    for i in datetime_list:
        order_dataframe[i] = pd.to_datetime(
            order_dataframe[i], format='%m/%d/%y %H:%M')

    # here the new column name needs to be inserted
    order_dataframe['created.month'] = order_dataframe['created_at'].dt.month
    order_dataframe['created.day'] = order_dataframe['created_at'].dt.dayofyear
    order_dataframe['created.year'] = order_dataframe['created_at'].dt.year


# Converts all entries to lowercase
def convert_lowercase_in_place(order_dataframe, string_list):
    for col in string_list:
        order_dataframe[col] = [
            str(x).lower() if x is not np.nan else np.nan for x in order_dataframe[col]]


def drop_cols_in_place(order_dataframe, droppable_list):
        # take in the dataframe and a list of column names that need to be dropped.
    # print(order_dataframe.columns)
    order_dataframe.drop(droppable_list, axis=1, inplace=True)


# Take in the dataframe combine/create categories
def merge_categories_in_place(order_dataframe):
    # converts categories into more descriptive strings and creates new column with numerical values for discount %
    order_dataframe['discount_code2'] = order_dataframe['discount_code'].astype(
        str).map(lambda x: x.lower())

    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(5off500|10off1000)', value='1%')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(10off500)', value='2%')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(15off1000)', value='1.5%')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(10%|add)(.*)', value='10%')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^.{12,}', value='other')
    order_dataframe['discount_perc'] = order_dataframe.discount_code.str.extract(
        '(\d+)', expand=True)
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(custom).(.*)', value='custom')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(mom)(.*)', value='mom')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(vday)(.*)', value='vday')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(winter)(.*)', value='winter')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(thankyou)(.*)', value='thankyou')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(fall)(.*)', value='fall')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(summer)(.*)', value='summer')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(cyber)(.*)', value='cyber')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(.*)(hat)(.*)', value='hat')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='^(flash)(.*)', value='flash')

    # after cleaning, reduce cateogories by merging codes with low frequency

    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(summer|winter|fall)', value='seasonal')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(xmas|easter|vday)', value='holiday')
    order_dataframe['discount_code2'].replace(
        inplace=True, regex=True, to_replace='(hat|prom|flash|cyber|funbag)', value='other')

    # merge and convert shipping methods
    order_dataframe['shipping_method2'] = order_dataframe['shipping_method']
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(fedex)(.*)(ground)(.*)', value='fedex_ground')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(free shipping|standard shipping)', value='fedex_ground')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(fedex)(.*)(overnight)(.*)', value='fedex_overnight')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(#)(.*)', value='nan')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(fedex 2 day)', value='fedex_2day')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(fedex express saver)', value='fedex_express')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(\d+)$', value='nan')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(ups)(.*)', value='ups')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(international)(.*)', value='fedex_international')
    order_dataframe['shipping_method2'].replace(
        inplace=True, regex=True, to_replace='(drop off at manhattan silver|store pickup)', value='nan')

    order_dataframe['shipping_continent'] = order_dataframe['shipping_country']

    order_dataframe['shipping_continent'].replace(
        inplace=True, regex=True, to_replace='(de|fr|gb|gp|nl)', value='europe')
    order_dataframe['shipping_continent'].replace(
        inplace=True, regex=True, to_replace='ca', value='canada')
    order_dataframe['shipping_continent'].replace(
        inplace=True, regex=True, to_replace='(bb|bm|do)', value='US')
    order_dataframe['shipping_continent'].replace(
        inplace=True, regex=True, to_replace='(au|ck|nz)', value='oceania')
    order_dataframe['shipping_continent'].replace(
        inplace=True, regex=True, to_replace='(cl|cr|mx|tt|uy)', value='americas')

    # merge and convert Payment Method
    order_dataframe['payment_method2'] = order_dataframe['payment_method']

    order_dataframe['payment_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(PayPal Express Checkout)(.*)', value='PayPal Express Checkout')
    order_dataframe['payment_method2'].replace(
        inplace=True, regex=True, to_replace='(Gift Card)(.*)', value='Gift Card')
    order_dataframe['payment_method2'].replace(
        inplace=True, regex=True, to_replace='(.*)(PayPal Express Checkout)(.*)', value='PayPal Express Checkout')
    # return new_order_dataframe


def improve_column_names(column_names):  # Remove spaces from the column names
    new_column_names = list(
        map(lambda x: x.lower().replace(' ', '_'), column_names))
    return new_column_names


def create_customer_id(order_dataframe):
    # column names need to be changed according to new column names, if applicable
    order_dataframe['customer_id'] = order_dataframe[[
        'shipping_zip', "shipping_company"]].apply(lambda values: '*'.join(values), axis=1)


def create_new_columns(order_dataframe):
    categories = ['ring', 'necklace', 'choker', 'earrings', 'scarf', 'studs', 'bracelet', 'chain', 'jewelry display', 'ears',
                  'collar', 'glitter post', 'keychain', 'bag', 'handbag', 'clip', 'hair', 'cover up', 'body jewelry', 'cuff', 'brooch',
                  'wallet', 'cap', 'sunglasses', 'hat', 'tiara', 'make-up remover', 'backpack', 'anklet', 'hairtie', 'drops', 'poncho',
                  'comb', 'kimono', 'shipping cost', 'rhinestone hoop', 'metal hoop', 'headband', 'fanny pack', 'pin', 'nails',
                  'mask', 'needles', 'decorative belt', 'purse', 'clutch', 'bangle', 'brush', 'sponge', 'ear threader', 'camisol',
                  'beanie', 'cream', 'gloves', 'cardigan', 'vest', 'shawl', 'texture hoop', 'watch', 'fish hook', 'charger', 'power bank',
                  'micro fan', 'fishhook']

    order_dataframe['category'] = order_dataframe['lineitem_name'].str.extract('({})'.format(
        '|'.join(categories)), flags=re.IGNORECASE, expand=False).str.lower().fillna('')
    order_dataframe['category'].replace('', 'other', inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['ears', 'studs', 'ear threader', 'texture hoop', 'metal hoop', 'rhinestone hoop', 'drops',
        'fish hook', 'needles', 'fishhook', 'clip'], 'earrings'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['clutch', 'bag', 'backpack', 'fanny pack', 'purse'], 'handbag'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['choker', 'chain', 'collar'], 'necklace'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['cuff', 'bangle', 'anklet'], 'bracelet'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['cover up', 'gloves', 'cardigan', 'vest', 'shawl', 'poncho','kimono', 'hat', 'scarf',
        'cap', 'beanie', 'camisol'], 'clothing'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['keychain', 'decorative belt','watch', 'wallet','sunglasses', 'body jewelry','glitter post',
        'brooch', 'tiara', 'hairtie','hair','headband', 'pin'], 'accessories'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['cream', 'nails','sponge','brush', 'mask', 'make-up remover'], 'cosmetics'), inplace=True)
    order_dataframe['category'].replace(dict.fromkeys(
        ['jewelry display', 'charger', 'power bank', 'micro fan', 'comb'], 'miscellaneous'), inplace=True)
    

    material = ['Onyx', 'Crystal', 'Rhinestone', 'Zirconia', 'Acetate', 'Precious', 'Glass', 'Stone', 'Diamond', 'Lucite', 'Metal', 'Acrylic',
                'Straw', 'Rhodium', 'Filigree', 'Pearl', 'Topaz', 'Leather', 'Fabric', 'Wooden', 'Mirror', 'Aluminium', 'Bamboo',
                'Embroidery', 'Wood', 'Semi-Precious', 'Stones', 'Fur', 'Snake', 'Reptile', 'Studded', 'Cotton', 'Suede', 'Garnet',
                'Geode', 'Woven', 'Amethyst', 'Abalone', 'Sapphire', 'Pebble', 'zirconia', 'Gem', 'Rhineston', 'Silk', 'Amber',
                'Emerald', 'Zircon', 'Aquamarine', 'Linen', 'Ruby', 'Viscose', 'Gold-Rhodium', 'Rhodium/Clear', 'Plastic', 'Steel',
                'Rhodium-Gold', 'Gemstones', 'Velvet', 'Rhodium-Pearl', 'Silicone', 'Rhodium-Sapphire',
                'Hematite-Silver', 'Hematite-Gold', 'Rhodium-Multi', 'Amethyst-Aurore', 'Crystal-Rhodium', 'Gold-Hematite-Brown',
                'Wooden-Fabric', 'Plaid']

    order_dataframe['material'] = order_dataframe['lineitem_name'].str.extract('({})'.format(
        '|'.join(material)), flags=re.IGNORECASE, expand=False).str.lower().fillna('')

    order_dataframe['material'].replace('', 'other', inplace=True)
    order_dataframe['material'].replace(
        'rhineston', 'rhinestone', inplace=True)
    order_dataframe['material'].replace('wood', 'wooden', inplace=True)

    shape = ['Cubic', 'Geometric', 'Skinny', 'Knot', 'Hemisphere', 'Linked', 'Oval', 'Circle', 'Round', 'Vertical', 'Arrowhead',
             'Rectangular', 'Oversized', 'Shaped', 'Cubes', 'Cube', 'Sphere', 'Stacked', 'Quintuple', 'Rectangle', 'Oblong', 'long',
             'Curve', 'Prolonged', 'Centered', 'Inverted', 'Cuboid', 'Triangular', 'Parallelogram', 'Pyramid', 'Fan-shaped', 'Angled', 'Octagonal', 'Doubled']

    order_dataframe['shape'] = order_dataframe['lineitem_name'].str.extract('({})'.format(
        '|'.join(shape)), flags=re.IGNORECASE, expand=False).str.lower().fillna('')

    order_dataframe['shape'].replace('', 'other', inplace=True)
    order_dataframe['lineitem_total'] = order_dataframe['lineitem_quantity'] * \
        order_dataframe['lineitem_price']

    color = ['aqua', 'beige', 'beige-mint', 'beige/black', 'beige/grey', 'beige/pink', 'berry', 'bk-brwon', 'bk-grey', 'bk-ivory',
             'bk-l.brown', 'bk-multi', 'black', 'black-5', 'black-brown', 'black-ivory', 'black-jet', 'black-matte', 'black-pearl',
             'black-red', 'black-silver', 'black-white', 'black/gold', 'black/grey', 'black/silver', 'black/white', 'blue', 'blue-7',
             'blue-ivory', 'blue-topaz', 'blue/yellow', 'brown', 'camouflage', 'champagne', 'cobalt', 'coral', 'coral-pink', 'cream',
             'crystal-gold', 'dark', 'floral', 'fuchsia', 'fuschia', 'gd-blue', 'gd-pink', 'gd-turquoise', 'gold', 'gold-6', 'gold-7',
             'gold-8', 'gold-aqua', 'gold-black', 'gold-blue', 'gold-brown', 'gold-gold', 'gold-gray', 'gold-peach', 'gold-pink',
             'gold-silver', 'gold-two', 'gold-worn', 'gold/black', 'gold/clear', 'gold/silver', 'gold/worn', 'gray', 'green', 'green-topaz',
             'grey', 'grey-2', 'grey/black', 'grey/mint', 'h-pink', 'herringbone', 'holographic', 'ivory', 'ivory-3', 'ivory-beige',
             'ivory-multi', 'ivory/beige', 'ivory/brown', 'ivory/brown-6', 'ivory/grey', 'khaki', 'l-pink', 'lavender', 'lavender/blue',
             'marble', 'metallic', 'mint', 'mint/opal', 'mint/pink', 'multi', 'multi-gold', 'mustard', 'navy', 'navy-red', 'neon', 'nude',
             'olive', 'olive/brown', 'orange', 'orange/pink', 'pastel', 'peach', 'peach-1', 'peach-grey', 'peach/blue', 'pink', 'pink-3',
             'pink-4', 'pink-beige', 'pink/brown', 'pink/burgundy', 'pink/coral', 'pink/mint', 'pink/opal', 'polka', 'purple', 'purple-blue',
             'rainbow', 'red', 'red-pink', 'rose', 'rosey', 'sandy', 'silver', 'silver-6', 'silver-7', 'silver-8', 'silver-brown',
             'silver-clear', 'silver-gold', 'silver-gray', 'silver-jet', 'silver-rustic', 'silver-turquoise', 'silver-worn',
             'sterling\xa0silver', 'teal', 'teal/brown', 'tone-brown', 'tortoise-1', 'turquoise', 'turquoise-blue', 'turquoise-pink',
             'turquoise/blue', 'turquoise/brown', 'turquoise/coral', 'two-tone', 'violet', 'watercolor', 'white', 'white-brown',
             'white-grey', 'white/brown', 'yellow', 'yellow/green']

    # order_dataframe['lineitem_name'] = order_dataframe['lineitem_name'].apply(lambda x: x.lower().split())

    def word_finder(str, words):
        return ([w for w in str if w in words])

    order_dataframe['color'] = order_dataframe['lineitem_name'].apply(
        lambda x: x.lower().split()).apply(lambda x: word_finder(x, color))

    new_color_columns = ['color1', 'color2', 'color3', 'color4']
    color_df = pd.DataFrame(
        order_dataframe['color'].values.tolist(), columns=new_color_columns)

    order_dataframe = pd.concat(
        [order_dataframe, color_df], axis=1, copy=False)

    order_dataframe.drop('color', axis=1, inplace=True)

    for color_col in new_color_columns:
        order_dataframe[color_col].fillna('None', inplace=True)
        order_dataframe[color_col].replace('gray', 'grey', inplace=True)
        order_dataframe[color_col].replace('grey/black', 'black/grey', inplace=True)
        order_dataframe[color_col].replace('bk-grey', 'black/grey', inplace=True)
        order_dataframe[color_col].replace('fuschia', 'fuchsia', inplace=True)
    order_dataframe['color1'].replace('None', 'other', inplace=True)


    # creating new features with different combinations
    order_dataframe['category*color1'] = order_dataframe[['category',
                                                          'color1']].apply(lambda value: '*'.join(value), axis=1)
    order_dataframe['category*color1*shape'] = order_dataframe[['category',
                                                                'color1', 'shape']].apply(lambda value: '*'.join(value), axis=1)
    order_dataframe['category*color1*shape*material'] = order_dataframe[['category',
                                                                         'color1', 'shape', 'material']].apply(lambda value: '*'.join(value), axis=1)

    order_dataframe['color1*shape'] = order_dataframe[['color1',
                                                       'shape']].apply(lambda value: '*'.join(value), axis=1)
    order_dataframe['color1*shaper*material'] = order_dataframe[['color1',
                                                                 'shape', 'material']].apply(lambda value: '*'.join(value), axis=1)

    order_dataframe['shape*material'] = order_dataframe[['shape',
                                                         'material']].apply(lambda value: '*'.join(value), axis=1)
    return order_dataframe


def clean_data(order_dataframe):  # Metafunction that calls all the cleaning functions

    # Remove spaces and convert to lowercase
    order_dataframe.columns = improve_column_names(order_dataframe.columns)
    # Replace "Name" feature with more accurate name "order_id"
    order_dataframe['order_id'] = order_dataframe['name']

    # Expand the "lineitem name" column into more descriptive features
    create_new_columns(order_dataframe)
    order_dataframe = create_new_columns(order_dataframe)

    # Convert all datetimes to datetime objects
    datetime_list = ["paid_at", "cancelled_at", "created_at"]
    convert_datetimes_in_place(order_dataframe, datetime_list)

    # Drop features that will not provide valuable information
    droppable_list = ["currency", "shipping_street", "shipping_address1",
                      "notes", "note_attributes", "source",
                      "tax_3_name", "name"]
    drop_cols_in_place(order_dataframe, droppable_list)

    # Convert all string entries to lower case
    string_list = order_dataframe.columns[order_dataframe.dtypes == object]
    convert_lowercase_in_place(order_dataframe, string_list)

    # Manually change certain entries to make fewer features
    merge_categories_in_place(order_dataframe)

    # Duplicate all order data for each purchased item
    desparsify_data_in_place(order_dataframe)

    # Make a unique customer ID from "Shipping Name" and "zip code"
    create_customer_id(order_dataframe)

    return order_dataframe
