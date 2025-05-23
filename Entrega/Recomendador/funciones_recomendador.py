import os
import pandas as pd
pd.options.display.max_columns = None
from IPython.display import display

import numpy as np
import plotly.express as px

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statistics import mean
from scipy import signal

from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from scipy.fft import fft

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dateutil.relativedelta import relativedelta
from datetime import datetime

import sys

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# -------------------- RECOMENDADOR----------------------------

def vectorizacion_descripcion_train(df_processed): 
    
    df_description = df_processed["tags"].to_list()
    
    # Vectorización del dataset (transformamos las frases en valores numéricos para poder alimentar al ML)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=500)
    df_description_numeric = vectorizer.fit_transform(df_description)
    df_description_numeric = df_description_numeric.toarray()
    
    
    return df_description_numeric, vectorizer

def vectorizacion_descripcion_val(df_processed, vectorizer): 
    
    df_description = df_processed["tags"].to_list()
    
    # Vectorización del dataset (transformamos las frases en valores numéricos para poder alimentar al ML)
    df_description_numeric = vectorizer.transform(df_description)
    df_description_numeric = df_description_numeric.toarray()
    
    return df_description_numeric

def modelo_semantico(df, division, valor, id_noticia, fecha):
    """Búsqueda semántica
    Fields:
        - Division = Region/País
        - Valor = valor de la región o país seleccionado
    """
    
    # Fecha a datetime
    fecha_dt = pd.to_datetime(fecha)
    
    # Restar 2 meses
    fecha_menos_2_meses = fecha_dt - relativedelta(months=2)

    # Quitamos aquellos que no tienen tags
    df["tags"] = df["tags"].fillna("")
    
    # Filtrado por id_noticia
    df_division = df[df[division] == valor]
    noticia = df_division[df_division["id"] == id_noticia]
    
    # Calculo de los embeddings
    df_description_numeric, vectorizer = vectorizacion_descripcion_train(df_division)
    
    # Calculo del embedding de la noticia
    noticia_processed_numeric = vectorizacion_descripcion_val(noticia, vectorizer)
    
    # Calculo de la distancia entre ambas
    cosine_similarities = cosine_similarity(noticia_processed_numeric, df_description_numeric)
    cosine_similarities = cosine_similarities[0]
    
    # Añadimos los embeddings
    df_division["embeddings"] = cosine_similarities
    
    # Realoizamos el filtrado
    df_filtrado = df_division[["id", "tags", "category", "published", division, "Vistas", "embeddings"]]
    
    # Filtrado por division
    df_filtrado = df_filtrado[df_filtrado["published"] >= fecha_menos_2_meses]
    print(f"Número de noticias: {len(df_filtrado)}")
    
    # Selección de los embeddings más cercanos
    df_filtrado = df_filtrado.sort_values('embeddings', ascending=False).head(200)
    df_filtrado = df_filtrado.sort_values('published', ascending=False).head(50)
    df_filtrado = df_filtrado[["id"]]
    
    # Asignamos valor de 50 a 0
    df_filtrado["score de relevancia"] =  range(50, 0, -1)
    
    return df_filtrado

def modelo_categoria(df, division, valor, id_noticia, fecha):
    """Modelo por categoría"""
    
    # Fecha a datetime
    fecha_dt = pd.to_datetime(fecha)
    
    # Restar 2 meses
    fecha_menos_2_meses = fecha_dt - relativedelta(months=2)

    
    # Selección del df por las columnas y filtrado por región
    df_division = df[df[division] == valor]
    df_filtrado = df_division[["id", "category", "subcategory", "published", division, "Vistas"]]
    
    # Filtrado por fecha
    df_filtrado = df_filtrado[df_filtrado["published"] >= fecha_menos_2_meses]
    print(f"Número de noticias: {len(df_filtrado)}")
    
    # Filtrado por id_noticia
    noticia = df_filtrado[df_filtrado["id"] == id_noticia]
    df_filtrado = df_filtrado[(df_filtrado["category"] == noticia["category"].values[0])]
    
    # Ordenamos por fecha
    df_filtrado = df_filtrado.sort_values('Vistas', ascending=False).head(200)
    df_filtrado = df_filtrado.sort_values('published', ascending=False).head(50)
    df_filtrado = df_filtrado[["id"]]
    
    # Asignamos valor de 50 a 0
    df_filtrado["score de relevancia"] =  range(50, 0, -1)
    
    
    return df_filtrado

def recomendacion(df, division, valor, id_noticia, fecha):
    
    
    # Búsqueda semántica
    noticias_ordenadas_semanticas = modelo_semantico(df, division, valor, id_noticia, fecha)
    
    # Búsqueda por categoria
    noticias_ordenadas_categoricas = modelo_categoria(df, division, valor, id_noticia, fecha)
    
    # Concatenamos resultados
    noticias_ordenadas_filtradas = pd.concat([noticias_ordenadas_categoricas, noticias_ordenadas_semanticas])
    
    # Agrupamos por id y sumamos valores
    df_result = noticias_ordenadas_filtradas.groupby('id')['score de relevancia'].sum().reset_index()
    df_result = df_result.sort_values('score de relevancia', ascending=False).head(6)
    
    # Quitamos la noticia original
    df_division = df[df[division] == valor]
    print("Noticia original:") # Replaced display with print
    display(df_division[df_division["id"] == id_noticia])
    df_division = df_division[df_division["id"] != id_noticia]
    
    # Obtenemos los títulos/información de las noticias
    df_result = df_division.merge(df_result, on=["id"], how="inner")
    df_result = df_result.sort_values('score de relevancia', ascending=False).head(5)
    
    return df_result

def main(df_pais, df_region):
    
    print("En caso de querer salir del programa, escribe: exit\n")

    # Variable
    continuacion = False
    while(continuacion == False):
        try:
            pregunta1 = input("Índicame el id de la noticia que estás viendo:  ")
            if(pregunta1 == "exit"):
                sys.exit(0)
            
            pregunta1 = np.int64(pregunta1)

            # Verificar si el id NO está en ninguno de los DataFrames
            id_en_df_pais = pregunta1 in df_pais["id"].values
            id_en_df_region = pregunta1 in df_region["id"].values

            if (not id_en_df_pais) and (not id_en_df_region):
                raise ValueError(f"\nEl ID de noticia '{pregunta1}' no se encontró en los datos.\n")

            # Si llegamos aquí, el ID fue encontrado en al menos uno de los DataFrames
            continuacion=True
            print(f"\nID de noticia ingresado: {pregunta1}\n")

        except Exception as e:
            print(f"\nError en la pregunta 1: {e}\n")
            
    # Variable
    continuacion = False
    
    while(continuacion == False):
        try:
            pregunta2 = input("Índicame la fecha en la que estás viendo la noticia en formato AÑO-MES-FECHA:  ")
            if(pregunta2 == "exit"):
                sys.exit(0)

            # Comprobación formato
            formato_esperado = "%Y-%m-%d"
            datetime.strptime(pregunta2, formato_esperado) # This will raise ValueError if format is wrong
            
            # If we reach here, format is correct
            continuacion = True
            print(f"\nLa fecha ingresada es: {pregunta2}\n")

        except ValueError as e: # Catch specific ValueError for datetime parsing
            print(f"\nError en el formato de la fecha: {e}. Por favor, usa el formato YYYY-MM-DD.\n")
        except Exception as e:
            print(f"\nError en la pregunta 2: {e}\n")
    
        if continuacion: # Only proceed with date range check if format was correct
            try:
                fecha_maxima = max(df_pais["published"].max(), df_region["published"].max())

                # Fecha a datetime
                fecha_maxima = pd.to_datetime(fecha_maxima)
                pregunta2_dt = pd.to_datetime(pregunta2)

                # Sumar 2 meses
                if (pregunta2_dt >= (fecha_maxima + relativedelta(months=2))):
                    # Correct variable name to avoid NameError if fecha_maxima_2meses wasn't defined
                    raise ValueError(f"\nNo existen datos para la fecha indicada. La fecha máxima de los datos es {fecha_maxima.strftime(formato_esperado)}.\n") 
                
            except Exception as e:
                print(f"\nError en la Pregunta 2 {e}\n")
                continuacion = False # Set continuacion to False to re-enter loop

    
    # Variable
    continuacion = False
    
    while(continuacion == False):
        
        try: 
            pregunta3 = input("¿De qué país eres?:    ")

            if(pregunta3 == "exit"):
                sys.exit(0)

            if(pregunta3.lower() == "españa"):
                pregunta4 = input("¿De qué región?:  ")

                if(pregunta4 == "exit"):
                    sys.exit(0)
                
                print(f"\nLa región ingresada es: {pregunta4}\n")
                
                # Verificar si la región NO está en ninguno de los DataFrames
                región_en_df_region = pregunta4 in df_region["Región"].values
                
                if (not región_en_df_region):
                    #raise ValueError(f"\nLa región de noticia '{pregunta4}' no se encontró en los datos.\n")
                    available_regions = sorted(df_region["Región"].dropna().unique().tolist())
                    raise ValueError(
                        f"\nLa región de noticia '{pregunta4}' no se encontró en los datos.\n"
                        f"Las regiones disponibles son: {available_regions}\n"
                    )
                
                # Recomendaciones
                df_result = recomendacion(df_region, "Región", pregunta4, pregunta1, pregunta2)
                continuacion = True
                

            else:
                
                print(f"\nEl país ingresado es: {pregunta3} \n")
                # Verificar si el país NO está en ninguno de los DataFrames
                pais_en_df_pais = pregunta3 in df_pais["País"].values
                
                if (not pais_en_df_pais):
                    #raise ValueError(f"\nEl país de noticia '{pregunta3}' no se encontró en los datos.\n")
                    available_pais = sorted(df_pais["País"].dropna().unique().tolist())
                    raise ValueError(
                        f"\nEl pais de noticia '{pregunta3}' no se encontró en los datos.\n"
                        f"Los paises disponibles son: {available_pais}\n"
                    )
                
                # Recomendaciones
                df_result = recomendacion(df_pais, "País", pregunta3, pregunta1, pregunta2)
                continuacion = True


        except Exception as e:
            print(f"\nError en la Pregunta 3 {e}\n")
        
        # Display the result outside the try-except block to ensure it's always displayed after a successful input
        if continuacion:
            print("Resultados de la recomendación:") # Replaced display with print
            display(df_result) # Replaced display with print
            
            return df_result
        
if __name__ == "__main__":
    main(df_noticias_pais, df_noticias_region)