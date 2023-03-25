# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# %%
'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

# %%
'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

# %%
def csv_to_df(path: str) -> pd.core.frame.DataFrame:
    return pd.read_csv(path)

#df_data = csv_to_df('StudentsPerformance.csv')
#df_data

# %%
'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

# %%
def capitalize_columns(df_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df_data_capitalized = df_data.copy()
    return df_data_capitalized.rename(columns=lambda columnName: columnName if "e" in columnName else columnName.upper())

#capitalize_columns(df_data)

# %%
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

# %%
def math_passed_count(df_data: pd.core.frame.DataFrame) -> int:
    df = df_data.copy()
    return len(df.loc[np.where(df['math score'] >= 50)].index)

#math_passed_count(df_data)

# %%
'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

# %%
def did_pre_course(df_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df_data.copy()
    return df.loc[np.where(df['test preparation course'] == 'completed')]

#did_pre_course(df_data)

# %%
'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

# %%
def average_scores(df_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df_data.copy()
    return df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean()

#average_scores(df_data)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''

# %%
def add_age(df_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df_data.copy()
    np.random.seed(42)
    df['age'] = np.random.randint(18, 67, df.shape[0]);
    return df

#add_age(df_data)


# %%
'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

# %%
def female_top_score(df_data: pd.core.frame.DataFrame) -> tuple:
    df = df_data.copy()
    df['sum_points'] = df['math score'] + df['reading score'] + df['writing score']
    df = df.loc[np.where(df['gender'] == 'female')].sort_values(['sum_points'], ascending=False)[['math score', 'reading score', 'writing score']].head(1)
    return (df.iloc[0]['math score'], df.iloc[0]['reading score'], df.iloc[0]['writing score'])

#female_top_score(df_data)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

# %%
def add_grade(df_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df = df_data.copy()
    df['grade'] = (df['math score'] + df['reading score'] + df['writing score']) / 300
    df['grade'] = df['grade'].apply(lambda grade: 'A' if grade >= 0.9 else 'B' if grade >= 0.8 else 'C' if grade >= 0.7 else 'D' if grade >= 0.6 else 'F')
    return df

#add_grade(df_data)

# %%
'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''

# %%
def math_bar_plot(df_data: pd.core.frame.DataFrame) -> mpl.figure.Figure:
    fig, ax = plt.subplots()
    ax.set_title('Average Math Score by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Math Score')

    df = df_data.copy()

    df = df.groupby('gender', as_index=False)['math score'].mean()

    ax.bar(df['gender'], df['math score'])

    return fig

#math_bar_plot(df_data)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''

# %%
def writing_hist(df_data: pd.core.frame.DataFrame) -> mpl.figure.Figure:
    fig, ax = plt.subplots()
    ax.set_title('Distribution of Writing Scores')
    ax.set_xlabel('Writing Score')
    ax.set_ylabel('Number of Students')

    df = df_data.copy()

    ax.hist(df['writing score'])
    return fig

#writing_hist(df_data)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

# %%
def ethnicity_pie_chart(df_data: pd.core.frame.DataFrame) -> mpl.figure.Figure:
    fig, ax = plt.subplots()
    ax.set_title('Proportion of Students by Race/Ethnicity')

    df = df_data.copy()
    df = df_data.groupby('race/ethnicity', as_index=False).size()

    ax.pie(df['size'], labels=df['race/ethnicity'], autopct='%1.1f%%')
    return fig

#ethnicity_pie_chart(df_data)



