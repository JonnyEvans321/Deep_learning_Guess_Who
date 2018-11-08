import pandas as pd
import numpy as np
import cv2    
import matplotlib.pyplot as plt
from PIL import Image
import math
import random
from pylab import show
import os
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model 
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

# set variables 
#define your local path here
main_folder=os.path.join(os.path.dirname(__file__))
images_folder = main_folder + '/data/celeba-dataset/img_align_celeba/'
attr_path='/data/celeba-dataset/list_attr_celeba.csv'

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
#just one epoch to save computing time, for more accurate results increase this number
NUM_EPOCHS = 1
#decide whether you want to see what the AI is doing
AI_PEEK=1

# import the data set that include the attribute for each picture
df_attr = pd.read_csv(main_folder + attr_path)
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0

## Import InceptionV3 Model
inc_model = InceptionV3(weights=main_folder+'/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
##Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# create the model 
model_ = Model(inputs=inc_model.input, outputs=predictions)

# Lock initial layers to do not be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                    , loss='categorical_crossentropy'
                    , metrics=['accuracy'])

#select 24 images randomly from the test set
df_to_test = df_attr.sample(24)

#make some unisex names up to give the game some character (the identity of the people in the images was anonymised)
char_names=['Alex', 'Ashley', 'Brooklyn', 'Bailey', 'Casey', 'Carson', 'Devon', 'Joe', 'Flynn', 'Finn', 'Haley', 'Jamie', 'Jude', 'Kayden', 'Kerry', 'Kim', 'Lee', 'Madison', 'Micah', 'Michel', 'Noel', 'North', 'Owen', 'Page']
random.shuffle(char_names)
#%%
#use some cool html to print out the prediction in a nice way
def make_prediction(filename,test_attr, target):

    im = cv2.imread(filename)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (178, 218)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)
    
    #first, load the relevant model weights
    model_.load_weights(main_folder+'/inceptionv3/attributes/weights.best.inc.'+test_attr+'.hdf5') 
    
    # prediction
    result = model_.predict(im)[0]
    
    #finds out if the ai is going to predict true or false
    prediction= True
    if result[1] <= 0.5:
        prediction = False
            
    return prediction

#make prediction
def ai_predict(test_attr):
    predictions=[]
    for index, name in enumerate(df_to_test.iterrows()):
        predictions.append(make_prediction(images_folder + df_to_test[test_attr].index[index],test_attr, df_to_test[test_attr][index]))
    return predictions

#plot grid of images       
def show_chars_grid(alive,player):
    #make grid
    fig,axes = plt.subplots(nrows = 4, ncols = 6, figsize=(20,10))
    
    #colour background of grid depending on whos go it is
    if(player=='AI'):
        col='red'
    else:
        col='green'
    fig.patch.set_facecolor(col)
    fig.patch.set_alpha(0.5)
    
    fig.suptitle(player+' celebrity board',fontsize=40)
    
    #take axis off each image
    for ax in axes.flatten():
        ax.axis('off')
    #make subplot for each image
    for index, name in enumerate(df_to_test.iterrows()):

        #set locatiion of where the image should be in the grid
        i=index%6
        j=math.floor(index/6)
        
        #get the image out
        image_path=images_folder + df_to_test.index[index]
        image = Image.open(image_path)
        axes[j,i].imshow(image)
        #add the 'name' of the celeb
        axes[j,i].set_title(char_names[index],fontsize=20,y=-.22)
        #add a cross to the celeb if they have been ruled out
        if(alive[index]=='REMOVED'):
            axes[j,i].scatter(0, 0, s=4000, c='red', marker='X')
    return plt

def show_players_char(imageid,name):
    #get a player image out
    image_path=images_folder + imageid
    image = Image.open(image_path)
    plt.axis('off')
    plt.title(name,fontsize=20)
    plt.imshow(image)
    return plt

def user_move(user_alive,result,AI_PEEK):
    #show grid of images
    show_chars_grid(user_alive,'Your')
    show(block=False)
    
    #lets the user take a guess at the AI's celeb
    print('Do you want to take a guess at the AI\'s celeb? Note that you can only guess the celeb once, if you get it wrong, you lose! (answer y or n)')
    while True:
        a=input()
        if(a=='y'):
            print('Please type the name of the celeb')
            while True:
                a=input()
                if(a==ai_name):
                    print('You win, thanks for playing!')
                    result=1
                    return user_alive,result
                elif(a in user_alive):
                    print('No, the AI\'s celeb is not ',a,', the AI wins!')
                    result=1
                    return user_alive,result
                else:
                    print('That is not an available celeb. Try again.')
                    continue
        elif(a=='n'):
            break
        else:
            print('\nPlease answer with y or n.')
            continue
        break
    
    #ask user to ask the AI whether its celeb has a certain attribute
    print('Available attributes:')
    print(user_unasked_attrs,)
    
    print('\nPick an attribute to ask the AI (enter the attribute):')
    while True:
        user_attr = input()
        if(user_attr in user_unasked_attrs):
            
            break
        else:
            if(user_attr in list(df_attr.columns)):
                
                print('\nYou\'ve already asked that. See the available attributes above.')
            else:
                print('\nThat\'s not an attribute. See the available attributes above.')
            continue
    
    #see if the ai thinks its celeb has the attribute or not
    ai_answer=ai_predictions[user_attr][char_names.index(user_name)]
    if(ai_answer==True):
        print('Yes, the AI thinks its character has',user_attr)
    else:
        print('No, the AI thinks its character doesn\'t have',user_attr)
    #show ai celeb
    if(AI_PEEK==1):
        print('AI\'s celeb:')
        show_players_char(ai_image,ai_name)
        show(block=False)
        
    show_chars_grid(user_alive,'Your')
    show(block=False)
    
    #user updates which celebs it could be
    print('Type in a celeb that you\'d like to remove from your grid (one at a time). When done, type \'done\'.')
    while True:
        char=input()
        if(char in user_alive):
            user_alive=[x if x!=char else 'REMOVED' for x in user_alive] 
            print('Done, remove another? (if not, type \'done\')')
            continue
        elif(char=='done'):
            return user_alive,result
        elif(char in char_names):
            print('You\'ve already removed them. Try again or type \'done\' to move on.')
            continue
        elif(char not in char_names):
            print('That\'s not a celeb name. Try again or type \'done\' to move on.')
            continue
    
    return user_alive,result

#AI's turn
def ai_move(ai_alive,result):
    
    #show the user their celeb so they can tell if they have the attribute
    print('Your celeb:')
    show_players_char(user_image,user_name)
    show(block=False)
    
    #AI finds the best attribute it could ask on this turn
    best_attr=''
    best_predictions=[]
    dist=24
    for attr in ai_unasked_attrs:
        #gather the AIs predictions for this attribute
        predictions=ai_predictions[attr]
        
        #we only want the predictions for the remaining celebs
        predictions=[predictions[i] for i in range(len(predictions)) if ai_alive[i] != 'REMOVED']
        
        #I think the best way the AI could pick a feature is to 'divide and conquer' ie find the feature that is closest to halves the ai's alive celebs
        if(abs(sum(predictions)-len(predictions)/2)<dist):
            dist=abs(sum(predictions)-len(predictions)/2)
            best_attr=attr
            
    #if the AI cant remove anymore celebs (because they all share the same remaining attributes or theres only one left), it makes a guess
    if(dist==len(predictions)/2):
        poss_celebs=[x for x in ai_alive if x!='REMOVED']
        print('AI: Is your celeb ',poss_celebs[0],'? (input y or n)')
        while True:
            a=input()
            if(a=='y'):
                print('The AI won, thanks for playing!')
                result=1
                return ai_alive,result
            elif(a=='n'):
                print('The AI guessed wrong, you win! thanks for playing!')
                result=1
                return ai_alive,result
                
            else:
                print('\nPlease answer with y or n.')
                continue
        
        
    #find the predictions of the best attribute over all celebs (just so they updating is easier)
    best_predictions=ai_predictions[best_attr]
    
    #update the list of attributes that can be asked
    ai_unasked_attrs.remove(best_attr)
    
    #ask user if their character has a particular attribute
    print('AI: Do they have '+best_attr+'? (input y or n) ')
    while True:
        a = input()
        if(a in ['y','n']):
            if(a=='y'):
                a=True
            else:
                a=False
            
            break
        else:
            print('\nPlease answer with y or n.')
            continue
    
    #update which characters remain
    for i in range(len(ai_alive)):
        if(ai_alive[i]!='REMOVED' and best_predictions[i]!=a):
            ai_alive[i]='REMOVED'
    
    return ai_alive,result
#%%
#ai predicts every attribute for the 24 celebs. Does this only once to save computation time
ai_predictions={}
for attr in list(df_attr.columns):
    ai_predictions[attr]=ai_predict(attr)
    print(attr)
#%%
#START GAME
#initialise attributes that havent been asked yet
ai_unasked_attrs=list(df_attr.columns)
user_unasked_attrs=list(df_attr.columns)
#intialise whether each character is still judged to be in the game
#XXX reword the above
user_alive=char_names.copy()
ai_alive=char_names.copy()
#flag to tell if the game has been won or not during a turn
result=0

#AI and user pick their celebs

#AI randomly chooses a character to be their own
rnd_no=random.randint(0,23)
ai_name=char_names[rnd_no]
ai_image=df_to_test.index[rnd_no]

#user gets the choice of which character to use
show_chars_grid(user_alive,'Your')
#this line makes matplotlib show the figure, rather than waiting until the rest of the script finishes
show(block=False)

#ask user to choose a celebrity to be their character
print('Please choose a celebrity (enter their name):')
while True:
    user_name = input()
    if(user_name in char_names):
        print('')
        break
    else:
        print('\nThat\'s not a celebrity, try again.')
        continue
user_image=df_to_test.index[char_names.index(user_name)]

print('Your celeb:')
show_players_char(user_image,user_name)
show(block=False)
if(AI_PEEK==1):
    print('AI\'s celeb:')
    show_players_char(ai_image,ai_name)
    show(block=False)

#play game
round_count=0
while True:
    round_count=round_count+1
    print('------- ROUND ',round_count,'-------')
    print('YOUR TURN')
    user_alive,result=user_move(user_alive,result,AI_PEEK)
    if(result==1):
        break
    """
    #show grid of images
    print('Your remaining celebs:')
    show_chars_grid(user_alive,'Your')
    show(block=False)
    """
    
    print('AI\'S TURN')
    ai_alive,result=ai_move(ai_alive,result)
    if(result==1):  
        break
    if(AI_PEEK==1):
        #show grid of images
        show_chars_grid(ai_alive,'AI')
        print('AI\'s remaining celebs:')
        show(block=False)
        