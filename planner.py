import pandas as pd
import matplotlib.pyplot as plt

data_path = "./kaggle/input/fitness-exercises-with-animations/fitness_exercises.csv"

exercise = pd.read_csv(data_path)

exercise.head(10)

exercise.shape

exercise.drop(['id'],axis=1,inplace = True)

exercise.rename(columns={'bodyPart':'muscleGroup'},inplace = True)

exercise.replace(to_replace = ['waist','cardio', 'upper arms', 'upper legs'],value=['core','cardiovascular', 'arms', 'legs'],inplace= True)


def getExercise(muscleGroup, df=exercise, numEx=1, includeEq=True):
    return df.loc[(df['muscleGroup'] == muscleGroup) & (includeEq | (df['equipment'] == 'body weight'))].sample(numEx)

if __name__ == "__main__":
    muscleGroup = input("What muscle would you like to work on? (Chest, core, arms, legs, back)\n").lower()
    numEx = int(input("How many exersices would you like?\n"))
    includeEq = input("Do you have access to equipment like weight and cables? (yes/no)\n").lower() == "yes"
    print(getExercise('chest', numEx=numEx, includeEq=includeEq))