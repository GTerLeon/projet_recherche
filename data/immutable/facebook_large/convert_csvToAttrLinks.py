import csv
#Used to transformed the facebook large page-page network target and edges files into attr and links files
#https://snap.stanford.edu/data/facebook-large-page-page-network.html
def transform_csv_to_links(input_file):
    output_file = input_file.split('.')[0] + '.links'

    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    transformed_data = [(row[0], row[1].replace(',', ' ')) for row in data]

    with open(output_file, 'w') as links_file:
        for id, value in transformed_data:
            links_file.write(f"{id} {value}\n")

    print(f"Transformation: {output_file}")


def transform_csv_to_attr(input_file):
    output_file = input_file.split('.')[0] + '.attr'

    with open(input_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    with open(output_file, 'w') as attr_file:
        for row in data:
            attr_file.write(f"{row[0]}\t{row[1]}\n")

    print(f"Transformation: {output_file}")

csv_nodes = "musae_facebook_target.csv"
csv_edges = "musae_facebook_edges.csv"

transform_csv_to_attr(csv_nodes)
transform_csv_to_links(csv_edges)   

