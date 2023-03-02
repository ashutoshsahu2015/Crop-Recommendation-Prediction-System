from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=UserWarning)

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method =='POST':
        
        ## Fetching the form data
        nitrogen = int(float(request.form['nitrogen']))
        phosphorous = int(float(request.form['phosphorous']))
        potassium = int(float(request.form['potassium']))
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        state = request.form['state']
        month = request.form['month']

        ## Season Variable

        try:
            if month in ['January','February']:
                season = 'Winter'
            elif month in ['March','April']:
                season = 'Spring'
            elif month in ['May','June']:
                season =  'Summer'    
            elif month in ['July','August']:
                season = 'Moonsoon'
            elif month in ['September','October']:
                season ='Autumn'
            elif month in ['November','December']:
                season = 'Pre Winter'

        except Exception as e:
            print("Creating Season variable went wrong")
            return(str(e))

        try:
            
            # Load the model from disk
            decision_recommendation = pickle.load(open('./model/DecisionTreerecommendation.pkl','rb'))
            randomforest_prediction = pickle.load(open('./model/RandomForestPredicition.pkl','rb'))
        
        except Exception as e:
            print("Error loading model")
            return str(e)

        try:
            # Load the label encoder model
            crop_label = pickle.load(open('./Prep/crop.pkl','rb'))
            month_label = pickle.load(open('./Prep/month.pkl','rb'))
            season_label = pickle.load(open('./Prep/season.pkl','rb'))
            state_label = pickle.load(open('./Prep/state.pkl','rb'))

        except Exception as e:
            print("Error Loading Label Encoder model")
            return(str(e))

        
        ## Recommendation Model
        
        try:
            X_test= pd.DataFrame({'nitrogen':nitrogen,'phosphorous':phosphorous,'potassium':potassium,'temperature':temperature,'humidity':humidity,'ph':ph,'rainfall':rainfall},index=[0])
            crop = decision_recommendation.predict(X_test)
            #print(crop)
        
        except Exception as e:
            print("Error in recommendation model")
            return(str(e))
        
        ## Dealing with categorical variables

        try:
            # Creating dataframe for category
            crop_df = pd.DataFrame({'crop':crop[0]},index=[0])
            month_df = pd.DataFrame({'month':month},index=[0])
            season_df = pd.DataFrame({'season':season},index=[0])
            state_df = pd.DataFrame({'state':state},index=[0])
        
            ## Transforming the category to label
            crop_label_encoded = crop_label.transform(crop_df)
            month_label_encoded = month_label.transform(month_df)
            season_label_encoded = season_label.transform(season_df)
            state_label_encoded = state_label.transform(state_df)

        except Exception as e:
            print("Error while dealing with category variables.")
            return(str(e))
        
        ## Prediction  Model
        price_information=[]
        try:
            for day in range(0,7):

                X_test = pd.DataFrame({'crop':crop_label_encoded,'state':state_label_encoded,'month':month_label_encoded,'season':season_label_encoded,'day':int(day)})
                price = randomforest_prediction.predict(X_test)

                if day == 0 :
                    weekday = 'Monday'
                elif day == 1:
                    weekday = 'Tuesday'
                elif day == 2:
                    weekday = 'Wednesday'
                elif day == 3:
                    weekday = 'Thursday'
                elif day == 4:
                    weekday = 'Friday'
                elif day == 5:
                    weekday = 'Saturday'
                else:
                    weekday = 'Sunday'
                
                price_details = {'day':weekday, 'price':str(round(price[0],2))}
                price_information.append(price_details)

        except Exception as e:
            print("Error in prediction model")
            return(str(e))
        
        return render_template('results.html',prices = price_information, crop_name = crop[0])

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)