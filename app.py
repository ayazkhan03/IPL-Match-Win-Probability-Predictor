import pickle
import pandas as pd
import streamlit as st

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali']

pipe  = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor 🏏')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team 🏏 ',sorted(teams))

with col2:
    bowlling_team = st.selectbox('Select the bowlling team ⚾',sorted(teams))

selected_city = st.selectbox('Select Host city 🏙️', sorted(cities))

target = st.number_input('Target 🎯')
col3,col4,col5 = st.columns(3)
with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs Completed')

with col5:
    wickets_out = st.number_input('Wickets Out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120-(overs*6)
    wickets = 10 - wickets_out
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowlling_team],
                  'city':[selected_city],'run_left':[runs_left],'ball_left':[balls_left],'wickets':[wickets],
                  'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    result = pipe.predict_proba(input_df)
    loss =result[0][0]
    win = result[0][1]
    st.header(batting_team + " - " + str(round(win*100)) + "% ✅")
    st.header(bowlling_team + " - " + str(round(loss*100)) + "% ❌")
