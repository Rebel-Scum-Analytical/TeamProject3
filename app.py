# Dependencies
import os
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd 
import sqlalchemy
import json
import decimal
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    make_response,
    session,
    abort,
    redirect,
    url_for,
    flash,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect, case
import pymysql
from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    BooleanField,
    TextField,
    PasswordField,
    SelectField,
    DateField,
    DecimalField,
    SubmitField
)
from wtforms.validators import InputRequired, Length, NumberRange, EqualTo
import datetime as dt
from decimal import Decimal
from Query_Visual import (
    createJson,
    creatUserPersonalJson,
    creatplotdata,
    CalculateDailyGoals,
)
import json
import plotly
import plotly.graph_objects as go

from dateutil import relativedelta
import dateutil.parser
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from food_recommendation import (
hillClimbing,
)
import redis
from rq import Queue
from worker import conn
from rq.job import Job

# redis_conn = redis.Redis(
#     host=os.getenv("REDIS_HOST", "127.0.0.1"),
#     port=os.getenv("REDIS_PORT", "6379"), 
#     password=os.getenv("REDIS_PASSWORD", ""),   
# )

# redis_queue = Queue(connection=redis_conn)

redis_queue = Queue(connection=conn)


#################################################
# Flask Setup
#################################################

app = Flask(__name__)

# Set the secret key value
app.secret_key = "1a2b3c4d5e"

#################################################
# Set up the database
#################################################
HOSTNAME = "127.0.0.1"
PORT = 3306
USERNAME = "root"
PASSWORD = "uv9y9g5t"
DIALECT = "mysql"
DRIVER = "pymysql"
DATABASE = "usda"

# Create a connection string to connect to DB in DB Server
db_connection_string = (
    f"{DIALECT}+{DRIVER}://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
)

# Database Setup for HEROKU

# db_cloud_string = os.getenv("JAWSDB_URL")

# app.config["SQLALCHEMY_DATABASE_URI"] = (
#     db_cloud_string )

app.config["SQLALCHEMY_DATABASE_URI"] = (
    os.environ.get("JAWSDB_URL", "") or db_connection_string
)

# databse setup for SQLAlchemy
db = SQLAlchemy(app)

# app.config["SQLALCHEMY_ECHO"] = True

# Create classes for the database tables and map the column names to all the database tables

# Meal_record table. This table is used to store data that user enters through URL to add meals
class Meal_record(db.Model):
    __tablename__ = "meal_record"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    meal_item_code = db.Column(db.Integer)
    username = db.Column(db.String(50))
    type = db.Column(db.String(50))
    meal_date = db.Column(db.Date)
    meal_desc = db.Column(db.String(256))
    amount = db.Column(db.Float)

    def __repr__(self):
        return "<Meal_record %r>" % (self.name)


# User_account table. THis table contains user profile information
class User_account(db.Model):
    __tablename__ = "user_account"

    username = db.Column(db.String(50), primary_key=True)
    password = db.Column(db.String(50))
    confirm_password = db.Column(db.String(50))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    gender = db.Column(db.String(50))
    date_of_birth = db.Column(db.Date)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    physical_activity_level = db.Column(db.String(50))

    def __repr__(self):
        return "<User_account %r>" % (self.name)


# Nutrition table. This table contains all the nutrition information for the food items in usda database.
class Nutrition(db.Model):
    __tablename__ = "nutrition"

    NDB_No = db.Column(db.Integer, primary_key=True, autoincrement=False)
    Shrt_Desc = db.Column(db.Text)
    Water = db.Column(db.Float)
    Energy = db.Column(db.Integer)
    Protein = db.Column(db.Float)
    Lipid_Total = db.Column(db.Float)
    Carbohydrate = db.Column(db.Float)
    Fiber = db.Column(db.Float)
    Sugar_Total = db.Column(db.Float)
    Calcium = db.Column(db.Integer)
    Iron = db.Column(db.Float)
    Magnesium = db.Column(db.Float)
    Phosphorus = db.Column(db.Integer)
    Potassium = db.Column(db.Integer)
    Sodium = db.Column(db.Integer)
    Zinc = db.Column(db.Float)
    Copper = db.Column(db.Float)
    Manganese = db.Column(db.Float)
    Selenium = db.Column(db.Float)
    Vitamin_C = db.Column(db.Float)
    Thiamin = db.Column(db.Float)
    Riboflavin = db.Column(db.Float)
    Niacin = db.Column(db.Float)
    Panto_Acid = db.Column(db.Float)
    Vitamin_B6 = db.Column(db.Float)
    Folate_Total = db.Column(db.Float)
    Folic_Acid = db.Column(db.Float)
    Food_Folate_mcg = db.Column(db.Float)
    Folate_DFE_mcg = db.Column(db.Float)
    Choline_Tot_mg = db.Column(db.Float)
    Vitamin_B12 = db.Column(db.Float)
    Vit_A_IU = db.Column(db.Integer)
    Vitamin_A = db.Column(db.Float)
    Retinol = db.Column(db.Float)
    Alpha_Carot_mcg = db.Column(db.Float)
    Beta_Carot_mcg = db.Column(db.Float)
    Beta_Crypt_mcg = db.Column(db.Float)
    Lycopene_mcg = db.Column(db.Float)
    Lut_Zea_mcg = db.Column(db.Float)
    Vitamin_E = db.Column(db.Float)
    Vitamin_D = db.Column(db.Float)
    Vit_D_IU = db.Column(db.Float)
    Vitamin_K = db.Column(db.Float)
    FA_Sat_g = db.Column(db.Float)
    FA_Mono_g = db.Column(db.Float)
    FA_Poly_g = db.Column(db.Float)
    Cholestrol = db.Column(db.Integer)
    Weight_grams = db.Column(db.Float)
    Weight_desc = db.Column(db.Text)
    GmWt_2 = db.Column(db.Float)
    GmWt_Desc2 = db.Column(db.Text)
    Refuse_Pct = db.Column(db.Integer)

    def __repr__(self):
        return "<Nutrition %r>" % (self.name)




# Initialize the data base and create tables
@app.before_first_request
def setup():
    db.create_all()

## Set up dataframe for recommendation models##
# Method to get the recommendation list of items similar to food items in advanced search
# filepath = "/db/nutrition.csv"
df = pd.read_csv("db/nutrition.csv")
print("Nutrition data is: ")
print(df.head())
X_text = df["Shrt_Desc"].values
cv = make_pipeline(
CountVectorizer(
    ngram_range=(3, 7),
    analyzer="char"),
    Normalizer())
cv.fit(X_text)
X = cv.transform(X_text)
## PK add code part1  to get term from advanced search box ##
# X_term = cv.transform(["choclte chip sookies"])
# simularities = cosine_similarity(X_term, X)

# Method to find recommendation for similar items
# Change the columns in dataframe declared above from butrition.csv to per calorie value
df["Protein/cal"] = df["Protein"] / df["Energy"]
df["Carbohydrtes/cal"] = df["Carbohydrate"] / df["Energy"]
df["Sodium/cal"] = df["Sodium"]/ df["Energy"]
df["Total_fat/cal"] = df["Lipid_Total"]/ df["Energy"]
df["Cholestrol/cal"] = df["Cholestrol"]/ df["Energy"]
df["Sugar/cal"] = df["Sugar_Total"]/ df["Energy"]
df["Calcium/cal"] = df["Calcium"]/ df["Energy"]    
df_percalorie = df[["NDB_No", "Shrt_Desc", "Carbohydrate", "Protein", "Lipid_Total", "Fiber", "Sugar_Total", "Protein/cal", "Carbohydrtes/cal", "Sodium/cal", "Sodium", 
"Total_fat/cal", "Cholestrol", "Sugar/cal", "Calcium/cal", "Calcium"]]
print("dataFrame per calorie value is: ")
print(df_percalorie.head())
# Removing null values from DataFrame
df_percalorie = df_percalorie.dropna(how='any',axis=0)
# Find the array for X values in recommendation model
X_nut = df_percalorie[['Protein/cal', 'Carbohydrtes/cal', 'Total_fat/cal', "Total_fat/cal", 'Sugar/cal']].values
# X_nut = df_percalorie[['Protein/cal', 'Carbohydrtes/cal', 'Sodium/cal', 'Cholestrol/cal', 'Sugar/cal', 'Calcium/cal']].values
X_norm = Normalizer().fit_transform(X_nut)

#############################################################################################
# Route #1("/")
# Home Page
#############################################################################################
@app.route("/index.html")
@app.route("/")
def main():
    session["page"] = " "
    # Check if the user is already logged in
    # If user is logged in then re-route to dashborad page
    if checkLoggedIn() == True:
        session["page"] = "dashboard"
        return redirect("/dashboard")
    # else route to home page
    session["page"] = " "
    return render_template("index.html")


#############################################################################################
# Route #2(/login)
# Design a query for the existing user to login
#############################################################################################


@app.route("/login", methods=["GET", "POST"])
def login():

    # Output message if something goes wrong...
    msg = ""
    if checkLoggedIn() == True:
        session["page"] = "dashboard"
        return redirect("/dashboard")
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == "POST":
        # Create variables for easy access
        request_username = request.form["username"]
        request_password = request.form["password"]

        # Check if account exists using MySQL
        if request_username and request_password:

            # Fetch one record and return result
            account = loginsys(request_username, request_password)

            # If account exists in accounts table in out database
            if account:
                # Create session data, we can access this data in other routes
                session["loggedin"] = True
                session["username"] = account[3]
                # Redirect to home page
                session["page"] = "dashboard"
                return redirect("/dashboard")
            else:
                # Account doesnt exist or username/password incorrect
                msg = "Incorrect username/password!"
    session["page"] = " "
    return render_template("index.html", msg=msg)


# Function to get the variable names used to validate the logged in user.
# This function retr=urns a user list with logged in user's first name, last name, gender and username
def loginsys(username, password):
    print("Username: " + username + " Password: " + password)
    user_ls = (
        db.session.query(
            User_account.first_name,
            User_account.last_name,
            User_account.gender,
            User_account.username,
        )
        .filter(User_account.username == username)
        .filter(User_account.password == password)
        .first()
    )
    print("user_ls: " + str(user_ls))
    return user_ls


##############################################################################################
# Route #3(/register)
# Design a query for the register a new user
#############################################################################################


class RegistrationForm(FlaskForm):
    username = StringField(
        "Username", validators=[InputRequired(), Length(min=4, max=20)]
    )
    password = PasswordField("Password", validators=[InputRequired()])
    confirm_password = PasswordField(
        "Confirm Password", validators=[InputRequired(), EqualTo("password")]
    )
    first_name = StringField(
        "First Name", validators=[InputRequired(), Length(min=2, max=50)]
    )
    last_name = StringField(
        "Last Name", validators=[InputRequired(), Length(min=2, max=50)]
    )
    gender = SelectField("Gender", choices=[("male", "Male"), ("female", "Female")])
    date_of_birth = DateField("Date of Birth (YYYY-MM-DD)", format="%Y-%m-%d")
    height = DecimalField(
        "Height (Inches)",
        places=2,
        rounding=None,
        validators=[InputRequired(), NumberRange(min=0, max=500, message="Blah")],
    )
    weight = DecimalField(
        "Weight (Pounds)",
        places=2,
        rounding=None,
        validators=[InputRequired(), NumberRange(min=0, max=2000, message="Blah")],
    )
    physical_activity_level = SelectField(
        "Physical Activity Level",
        choices=[
            ("sedentary", "Sedentary"),
            ("lightly active", "Lightly Active"),
            ("moderately active", "Moderately Active"),
            ("very active", "Very Active"),
            ("extra active", "Extra Active"),
        ],
    )
    submit = SubmitField("Get Started")


@app.route("/register", methods=["GET", "POST"])
def register():
    if checkLoggedIn() == True:
        session["page"] = "dashboard"
        return redirect("/dashboard")

    form = RegistrationForm(request.form)
    if form.validate_on_submit():
        flash(f"Account created for {form.username.data}!", "success")

        new_user = User_account(
            username=form.username.data,
            password=form.password.data,
            confirm_password=form.confirm_password.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            gender=form.gender.data,
            date_of_birth=form.date_of_birth.data,
            height=form.height.data,
            weight=form.weight.data,
            physical_activity_level=form.physical_activity_level.data,
        )
        db.session.add(new_user)
        db.session.commit()

        return redirect("/dashboard")
    return render_template("New_user.html", form=form)


########################################################################################################
# Route #4(/dashborad)
# Design a query to display dashboard to the logged in user.
# This route will provide the following functionalities and will be available for only logged in users.
# 1) Display daily statistics to the logged in user.
# 2) Provide a quick add feature where logged in user can add his meals.
# 3) Dashboard will display last 5 entries that the user made using quick add mentioned in point above.
# 4) This will add the meal information that user enters into database
########################################################################################################
def getUserpersonalData(user):
    cmd1 = db.session.query(
        User_account.height.label("height"),
        User_account.weight.label("weight"),
        User_account.physical_activity_level.label("phy"),
        User_account.gender.label("gender"),
        User_account.date_of_birth.label("dob"),
    ).filter(User_account.username == user)
    user_info = cmd1.first()
    return creatUserPersonalJson(user_info)


class AddMeal(FlaskForm):
    inputdate = DateField("inputdate", format="%Y-%m-%d")
    meal_category = StringField(
        "meal_category", validators=[InputRequired(message="Meal type is required")]
    )
    food_desc = StringField(
        "food_desc", validators=[InputRequired(message="Search the food item")]
    )
    servings_count = DecimalField(
        "servings_count",
        places=2,
        rounding=None,
        validators=[
            InputRequired(message="Serving count is required"),
            NumberRange(min=0.25, max=20),
        ],
    )
    foodNameId = StringField("foodNameId")
    submit = SubmitField("Add")


# Code to display daily statistics on dashboard
# daily_goal_list = [1800, 130, 25, 2200, 25, 25.2]


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if checkLoggedIn() == False:
        return redirect("/login")

    session["page"] = "dashboard"
    msg = ""
    # Code to display daily statistics on dashboard - part 1

    # code added to get the Daily goal as per the user gender, height, weight, age and physical activity
    session_user_name = session["username"]
    user_personal_data = getUserpersonalData(session_user_name)
    daily_goal_list = CalculateDailyGoals(user_personal_data)
    print(daily_goal_list)

    form = AddMeal(request.form)
    if form.validate_on_submit():

        # flash(f'Meal Added for {form.meal_category.data}!', 'successfully')

        new_meal = Meal_record(
            username=session["username"],
            meal_date=form.inputdate.data,
            type=form.meal_category.data,
            meal_desc=form.food_desc.data,
            amount=form.servings_count.data,
            meal_item_code=form.foodNameId.data,
        )
        # if new_meal.type is null:
        #     print("Please enter a valid meal type value")
        #     msg = "Please enter a valid value"
        # elif new_meal.meal_desc is null:
        #     print("Please search a meal item")
        #     msg = "Please enter a valid value"
        # elif new_meal.amount is null:
        #     print("Please enter a valid serving size")
        #     msg = "Please enter a valid serving sevalue"

        # else:
        db.session.add(new_meal)
        db.session.commit()
        flash("Meal saved successfully!")
        print("Adding meal")
        return redirect("/dashboard")

    # Code to display daily statistics on dashboard - part 2
    # display_stats
    cmd = (
        db.session.query(
            func.round(
                func.coalesce(
                    func.sum(
                        (Nutrition.Energy / 100)
                        * (Meal_record.amount)
                        * (
                            case(
                                [(Nutrition.Weight_grams == 0, 100)],
                                else_=Nutrition.Weight_grams,
                            )
                        )
                    ),
                    0,
                ),
                2,
            ).label("cal"),
            func.round(
                func.coalesce(
                    func.sum(
                        (Nutrition.Carbohydrate / 100)
                        * (Meal_record.amount)
                        * (
                            case(
                                [(Nutrition.Weight_grams == 0, 100)],
                                else_=Nutrition.Weight_grams,
                            )
                        )
                    ),
                    0,
                ),
                2,
            ).label("carbs"),
            func.round(
                func.coalesce(
                    func.sum(
                        (Nutrition.Protein / 100)
                        * (Meal_record.amount)
                        * (
                            case(
                                [(Nutrition.Weight_grams == 0, 100)],
                                else_=Nutrition.Weight_grams,
                            )
                        )
                    ),
                    0,
                ),
                2,
            ).label("protein"),
            func.round(
                func.coalesce(
                    func.sum(
                        ((Nutrition.Sodium) / 100)
                        * (Meal_record.amount)
                        * (
                            case(
                                [(Nutrition.Weight_grams == 0, 100)],
                                else_=Nutrition.Weight_grams,
                            )
                        )
                    ),
                    0,
                ),
                2,
            ).label("sodium"),
            func.round(
                func.coalesce(
                    func.sum(
                        (
                            (Nutrition.Water / 1000) / 100
                        )  # we are divding by 1000 to convert ml to Liters
                        * (Meal_record.amount)
                        * (
                            case(
                                [(Nutrition.Weight_grams == 0, 100)],
                                else_=Nutrition.Weight_grams,
                            )
                        )
                    ),
                    0,
                ),
                2,
            ).label("water"),
            func.round(
                func.coalesce(
                    func.sum(
                        (Nutrition.Fiber / 100)
                        * (Meal_record.amount)
                        * (
                            case(
                                [(Nutrition.Weight_grams == 0, 100)],
                                else_=Nutrition.Weight_grams,
                            )
                        )
                    ),
                    0,
                ),
                2,
            ).label("fiber"),
            func.count().label("cnt"),
        )
        .filter(Meal_record.username == session["username"])
        .filter(Meal_record.meal_item_code == Nutrition.NDB_No)
        .filter(Meal_record.meal_date == dt.date.today())
    )
    # print ("daily_total qry: "+ str(cmd))
    daily_stats = cmd.first()

    results = [0.0, 0, 0, 0, 0, 0]

    if daily_stats.cnt != 0:
        results = [
            float(daily_stats.cal),
            float(daily_stats.carbs),
            float(daily_stats.protein),
            float(daily_stats.sodium),
            float(daily_stats.water),
            float(daily_stats.fiber),
        ]

    print("daily stats are: ", daily_stats)
    print("daily stats cnt: ", daily_stats.cnt)

    # Code to display last 5 entries on dashboard
    top5_entries = (
        db.session.query(Meal_record)
        .filter(Meal_record.username == session["username"])
        .order_by(Meal_record.id.desc())
        .limit(5)
    )
    print("Top 5 entries are: ", top5_entries)
    return render_template(
        "dashboard.html",
        form=form,
        results=results,
        daily_goal_list=daily_goal_list,
        top5_entries=top5_entries,
        daily_stats=daily_stats,
        msg=msg,
    )


##############################################################################################
# Route #5(/intake)
# To display meal entries made by user in meal add section under dashboard section.
# The meal entries are displayed in a table form under food history
#############################################################################################
@app.route("/intake")
def food_tracker():
    if checkLoggedIn() == False:
        return redirect("/login")

    session["page"] = "intake"
    # Query to display last 100 entries on food_diary
    top100_entries = (
        db.session.query(Meal_record)
        .filter(Meal_record.username == session["username"])
        .order_by(Meal_record.meal_date.desc())
        .limit(100)
    )
    

    return render_template("food_history.html", top100_entries=top100_entries)

# Heroku background task status
def get_status(job):
    status = {
        'id': job.id,
        'result': job.result,
        'status': 'failed' if job.is_failed else 'pending' if job.result == None else 'completed'
    }
    status.update(job.meta)
    return status

# @app.route("/intake")
# def intake():
#     if checkLoggedIn() == False:
#         return redirect("/login")
#     session["page"] = "intake"
#     return render_template("intake.html")

##############################################################################################
# Route #6(/analysis)
# Design a query to display daily visualisations of the food intake by the user
#############################################################################################


@app.route("/analysis", methods=["GET", "POST"])
def analysis():


    global deficient_nutrients
    global displaylist
    global target_nutrients_corrected

    if checkLoggedIn() == False:
        return redirect("/login")
    session["page"] = "analysis"

    # plot_type = request.args.get("selectnutrients")
    plot_type = "All"
    desired_date = request.args.get("date")

    end_date = request.args.get("enddate")

 
    

    if request.method == "GET" and desired_date :
        print(f"desired date : {desired_date}")
        print(f"end date : {end_date}")
        starting_date = dateutil.parser.parse(desired_date)
        ending_date =  dateutil.parser.parse(end_date)

        # plus one to include start and end dates into num_days
        num_days= (relativedelta.relativedelta(ending_date, starting_date).days)+1
        

        cmd = (
            db.session.query(
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Energy / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("cal"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Water / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("water"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Carbohydrate / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("carbs"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Fiber / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("fiber"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Protein / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("protein"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Calcium / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("calcium"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Copper / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("copper"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Iron / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("iron"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Magnesium / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("magnesium"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Manganese / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("manganese"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Phosphorus / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("phosphorus"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Selenium / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("selenium"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Zinc / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("zinc"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Potassium / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("potassium"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Sodium / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("sodium"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_A / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_A"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_C / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_C"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_D / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_D"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_E / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_E"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_K / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_K"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Thiamin / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("thiamin"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Riboflavin / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("riboflavin"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Niacin / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("niacin"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_B6 / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_B6"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Folate_Total / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("folate"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Vitamin_B12 / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("vitamin_B12"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Panto_Acid / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("panto_acid_VB5"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Choline_Tot_mg / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("choline"),
                func.round(
                    func.coalesce(
                        func.sum(
                            (Nutrition.Lipid_Total / 100)
                            * (Meal_record.amount)
                            * (Nutrition.Weight_grams)
                        ),
                        0,
                    ),
                    2,
                ).label("fats"),
            )
            .join(Meal_record, Nutrition.NDB_No == Meal_record.meal_item_code)
            .filter(Meal_record.username == session["username"])
            .filter((Meal_record.meal_date >= desired_date),(Meal_record.meal_date <= end_date))
        )

        nutri_stats = cmd.first()

        userdata_nutrition_data = createJson(nutri_stats)
        session_user_name = session["username"]
        user_personal_data = getUserpersonalData(session_user_name)

        user_info = {
            "userdata_nutrition_data": userdata_nutrition_data,
            "user_personal_data": user_personal_data,
            "plot_type": plot_type,
        }

        return_list = creatplotdata(user_info,num_days)
        graphJSON = return_list[0]
        deficient_nutrients = return_list[1]
        displaylist = return_list[2]
        target_nutrients_corrected= return_list[3]

        plot_ids = ["plot1", "plot2", "plot3"]

        return render_template(
            "Daily_vizualization.html", plot_ids=plot_ids, graphJSON=graphJSON, date=desired_date ,  enddate=end_date

        )
    if request.method == "POST":
        if(len(deficient_nutrients)):
            input_to_function = {"first":deficient_nutrients,
            "second":displaylist,
            "third":target_nutrients_corrected,
            "fourth":5
            }
            job = redis_queue.enqueue(hillClimbing,input_to_function, job_timeout=600)
            tables = None
            job_id=job.get_id()
            data={'Location': url_for('job_status', job_id=job.get_id())}
            print(json.dumps(job_id))
            return render_template("food_reco.html", tables=tables, job_id=json.dumps(job_id))
        else:
            tables = None
            return render_template("food_reco.html", tables=tables, job_id=None)
    
    return render_template("Daily_vizualization.html")

# Function to check if the user is logged in and maintain the infomration in session variable.
# This is used in multiple routes.
def checkLoggedIn():
    if "loggedin" in session:
        if session["loggedin"] == True:
            return True
    return False


##################################################################################################
# Route #7(/nutrition)
# "Nutrition Lookup"
# To display nutrition information in table form for the searched food item on "Nutrition Lookup"
# This will provide a search window where user can enter text and will return a clickable list
# of matchig food entries. When user selects any item from the list, the code displays nutrition
# information for the selected food item
##################################################################################################
# Recommendation model2 -otems similar in composition ##
# Create a method to recommend 5 items similar to text in search string
def similar_items(term):
    idx = int(df_percalorie[df_percalorie['NDB_No'] == int(term)].index.values)
    # idx = idx[0]
    similarities = cosine_similarity(X_norm[idx].reshape(1,-1), X_norm)
    k = 5
    result = np.sort(np.argpartition(similarities[0], len(similarities[0]) - k)[-k:])
    print("list of similar items: ")
    print("result for similar items:")
    print(df_percalorie.iloc[result].columns)
    print(df_percalorie.iloc[result].head())
    return df_percalorie.iloc[result].values.tolist()
    # return

@app.route("/nutrition", methods=["GET"])
def nutrition():
    if checkLoggedIn() == False:
        return redirect("/login")
    session["page"] = "nutrition"

    ndbNo = request.args.get("ndbNo")

    if ndbNo:
        nutriData = (
            db.session.query(Nutrition).filter(Nutrition.NDB_No == ndbNo).first()
        )

        similarResult = similar_items(ndbNo)
        print('SimilarResult')
        print(similarResult)
        return render_template("nutrition.html", nutriData=nutriData,similarResult=similarResult)
    return render_template("nutrition.html")


##################################################################################################
# Route #8(/logout)
# Design a query for the existing user to logout.
##################################################################################################
@app.route("/logout")
def logout():
    if checkLoggedIn() == False:
        session["page"] = " "
        return render_template("login.html", msg="Already logged out!")
    else:
        session["loggedin"] = False
        messages = "loggedout"
        session["messages"] = messages
        session["page"] = " "
        return redirect("/")
    return "/"


######################################################################################################
# Route #9(/nutriquicksearch)
# Design a query for the existing user to search food item and display a list of matcing items.
# This will display a list of matching records and user can select the food item of their choice.
# The list will display the food item name along with the item weight in grams and weight description.
######################################################################################################
class DecimalEncoder(json.JSONEncoder):



    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)


@app.route("/nutriquicksearch", methods=["GET"])
def nutriquicksearch():
    if checkLoggedIn() == False:
        return redirect("/login")
    searchkey = request.args.get("term")
    if not searchkey:
        return '{  "data": [] } '
    resultSet = (
        db.session.query(
            Nutrition.NDB_No,
            Nutrition.Shrt_Desc,
            Nutrition.Weight_desc,
            Nutrition.Weight_grams,
        )
        .filter(Nutrition.Shrt_Desc.ilike("%" + searchkey + "%"))
        .all()
    )
    return json.dumps(resultSet, cls=DecimalEncoder)


##################################################################################################
# Route #10(/profile)
# Design a query for display the profile information for the logged in user.
##################################################################################################
@app.route("/profile")
def profile():
    if checkLoggedIn() == False:
        return redirect("/login")

    session["page"] = "profile"
    # Query to display the user profile
    user_profile = (
        db.session.query(User_account)
        .filter(User_account.username == session["username"])
        .all()
    )

    # print("User profile is: ", user_profile)

    return render_template("/profile.html", user_profile=user_profile)

######################################################################################################
# Route #11(/advanced_search)
# Design a query for display the items similar to the text entered in search box - "Advanced search"
######################################################################################################
## Recommendation model1 - advance search ##
def advanced_search_func(term):
    X_term = cv.transform([term])
    simularities = cosine_similarity(X_term, X)
    k = 10
    result = np.sort(np.argpartition(simularities[0], len(simularities[0]) - k)[-k:])
    return df.loc[result][['NDB_No', 'Shrt_Desc', 'Weight_desc', 'Weight_grams']]

@app.route("/advanced_search",methods=["GET"])
def advanced_search():
    if checkLoggedIn() == False:
        return redirect("/login")
    # session["page"] = "dashboard"  
    # Get the term from advanced dearch bar on dashoard page
    term = request.args.get("term")
    if not term:
        return '{  "data": [] } '
    ## Add logic here  to find the search term ##
    print('term: '+term)
    searchResult = advanced_search_func(term)
    return json.dumps(searchResult.values.tolist(), cls=DecimalEncoder)
    #return(json.dumps(str(searchResult.values.tolist())))

######################################################################################################
# Route #12(/job_status)
# This route is to start the background task for Food recommendation based on the nutrition
######################################################################################################

@app.route("/job_status/<job_id>")
def job_status(job_id):
    
    job = Job.fetch(job_id, connection=conn)
    if job is None:
        response = {'status': 'unknown'}
    else:
        response = {
            'status': job.get_status(),
            'result': job.result,
        }
        print(job.result)
        print(job.get_status())
        if job.is_failed:
            response['message'] = job.exc_info.strip().split('\n')[-1]
        if job.get_status() == "finished":
            print("I am inside finished")
            basket_NDB = job.result
            lastelement = len(basket_NDB.index)
            basket_NDB.index = pd.RangeIndex(start=1,stop=(lastelement+1), step=1)

            basket_NDB = basket_NDB.drop(['NDB_No'], axis=1)
            basket_NDB = basket_NDB.rename(columns={'Shrt_Desc': 'Food'})
            basket_NDB_Transpose = basket_NDB.T
            basket_NDB_Transpose = basket_NDB_Transpose.add_prefix('Entry_')
            # jsonfiles = json.dumps(basket_NDB_Transpose.values.tolist(), cls=DecimalEncoder)

            tables=[basket_NDB_Transpose.to_html(classes='table table-dark', table_id ='diary-table', justify='center')]
            # titles=basket_NDB_Transpose.columns.values
            # return render_template("food_reco.html", tables=tables)
            response = {
            'status': job.get_status(),
            'result': tables,
            }


    return jsonify(response)

######################################################################################################
# Route #12(/Background_process)
# This route is to start the background task for Food recommendation based on the nutrition
######################################################################################################

@app.route("/background_process")
def background_process():

    
    print(f"Decificent Nutrients : {deficient_nutrients}")

    if(len(deficient_nutrients)):
        input_to_function = {"first":deficient_nutrients,
        "second":displaylist,
        "third":target_nutrients_corrected,
        "fourth":5
        }
        job = redis_queue.enqueue(hillClimbing,input_to_function, job_timeout=600)
        print(f" Job ID : {job.id}")

        return jsonify({}), 202, {'Location': url_for('job_status', job_id=job.get_id())}
        # return jsonify({"job_id": job.id})
    else:
        return jsonify({}), 202, {'Location': url_for('job_status', job_id=null)}
            # return jsonify({"job_id": null})
        # return jsonify(output)
        # data_to_display = pd.DataFrame(columns=["Message"],data=[ "Please wait while the recommendation is processed"])                
        # tables = [data_to_display.to_html(classes='table table-dark', table_id ='diary-table', justify='center')]
    
    








if __name__ == "__main__":
    app.run(debug=True)
    # app.run()
