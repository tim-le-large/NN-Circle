# training data for the neural network with two inputs and one output
import csv
import random


class Data:

    @staticmethod
    def is_coordinate_in_circle(x, y):
        return x ** 2 + y ** 2 < 1

    def generate_circle_data(self, num_of_data):
        data = []
        for i in range(num_of_data):
            x_value = random.uniform(-2, 2)
            y_value = random.uniform(-2, 2)
            target_value = 0.8 if self.is_coordinate_in_circle(x_value, y_value) else 0
            data.append([x_value, y_value, target_value])
        return data

    def generate_circle_data_csv(self, num_of_data):
        fields = ['X', 'Y', 'TARGET']
        rows = []
        for i in range(num_of_data):
            x_value = random.uniform(-2, 2)
            y_value = random.uniform(-2, 2)
            target_value = 0.8 if self.is_coordinate_in_circle(x_value, y_value) else 0
            row = [x_value, y_value, target_value]
            rows.append(row)
        filename = "circle_data.csv"
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the fields
            csvwriter.writerow(fields)
            # writing the data rows
            csvwriter.writerows(rows)
