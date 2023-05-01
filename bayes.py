import readdata as rd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def NaiveBayes(pixel_count, prior_prob,test_image, image_count, type): # 28 rows of an image, pixel count has the number of pixels for each row for each number image (1-9)
    if type == "Digits":
        likelihood = [1] * 10
    elif type == "Faces":
        likelihood = [1] * 2
    
    for j in range(0, len(likelihood)):
        prior_value = prior_prob[j]
        for i in range(0,len(test_image)):
            test_value = test_image[i]
            if pixel_count[j,i,test_value] == 0:
                likelihood[j] = likelihood[j] * (1 / image_count[j])
            else:
                num_of_occ = pixel_count[j, i, test_value]
                likelihood[j] = likelihood[j] * (num_of_occ / image_count[j])
        
        
        likelihood[j] = likelihood[j] * prior_value
    return likelihood.index(max(likelihood))
    
            

def algo(Y_train_labels, X_train, X_test, Y_test_labels, type):
        if type == "Digits":
            classes = 10
            image_length = 28
            image_width = 28
            num_of_images = 1000
            ran_state = 30
        elif type == "Faces":
            classes = 2
            image_length = 70
            image_width = 60
            num_of_images = 150
            ran_state = 10

        tem = 1
        accuracy_array = []
        percent_training_labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        total_training_time = 0
        total_start = time.time()

        for i in range(0, 10): #Numbers range from 0 to 9
            start = time.time()
            tem -= 0.10
            if tem < 0:
                tem = 0.0001
            x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train_labels, test_size=tem, random_state=ran_state)

            
            pixel_count = np.zeros((classes, image_length, image_width))
            image_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            for w in range(0, len(x_train)):
            
                #Checks what number image is via labels 
                k = int(y_train[w])
                image_count[k] = image_count[k] + 1
                it = -1
                #x_train[w] is 28 by 28 and i of x_train[w] is 28 rows (Goes through each image and populates the pixel matrix)
                for i in x_train[w]:
                    it = it + 1
                    count = 0
                    for j in i:
                        if j == 1:
                            count = count + 1
                    pixel_count[k, it, count] += 1

        #  print(image_count) Image is a 1 by 10 matrix
            
        

            prior = classes * [0]
            for i in range(0, classes):
                num_of_labels = len(y_train)
                prior[i] = image_count[i] / num_of_labels

            end = time.time()
            total_training_time = total_training_time + (end - start)
            img_di = image_length * [0]
            predicted_values = []
            for w in range(0, num_of_images): #Number of images in test set is always 1000
                count = 0
                it = 0
                for i in X_test[w]:
                    count = 0
                    for j in i:
                        if j == 1:
                            count = count + 1
                    img_di[it] = count
                    it = it + 1
                
                # img_di Tells how many pixels each row of an image has 
                
                pred_lab = NaiveBayes(pixel_count, prior, img_di, image_count, type)
                predicted_values.append(pred_lab)

        
        
            final_count = 0
            for i in range(0, num_of_images): #Number of labels in test will always be 1000
                if int(Y_test_labels[i]) == predicted_values[i]:
                    final_count += 1

            
            print("Accuracy with "+str(round(1-tem,2)*100) + "% training data: " + str(final_count/len(Y_test_labels)*100))
            accuracy_array.append(final_count / len(Y_test_labels) * 100)



        total_end= time.time()
        plt.plot(percent_training_labels, accuracy_array, '-ko', linewidth=1, markersize=3)
        plt.xlabel("Partition Percentage")
        plt.ylabel("Accuracy")
        plt.show()

        print("Total training time: ", total_training_time, " seconds")
        print("Total time taken: ", total_end - total_start, " seconds")


train_images = "./data/digitdata/trainingimages"
train_labels = "./data/digitdata/traininglabels"
test_images = "./data/digitdata/testimages"
test_labels = "./data/digitdata/testlabels"



get_data_train = rd.load_data(train_images, 5000, 28, 28)
get_data_test = rd.load_data(test_images, 1000, 28, 28)
Y_train_labels = labels = rd.load_label(train_labels,5000)
X_train = rd.matrix_transformation(get_data_train, 28, 28)
X_test = rd.matrix_transformation(get_data_test, 28, 28)
Y_test_labels = rd.load_label(test_labels,1000)

algo(Y_train_labels, X_train, X_test, Y_test_labels, "Digits")



train_images2 = "./data/facedata/facedatatrain"
train_labels2 = "./data/facedata/facedatatrainlabels"
test_images2 = "./data/facedata/facedatatest"
test_labels2 = "./data/facedata/facedatatestlabels"



get_data_train2 = rd.load_data(train_images2, 451, 70, 60)
get_data_test2 = rd.load_data(test_images2, 150, 70, 60)
Y_train_labels2 = labels = rd.load_label(train_labels2,451)
X_train2 = rd.matrix_transformation(get_data_train2, 70, 60)
X_test2 = rd.matrix_transformation(get_data_test2, 70, 60)
Y_test_labels2 = rd.load_label(test_labels2,150)

algo(Y_train_labels2, X_train2, X_test2, Y_test_labels2, "Faces")