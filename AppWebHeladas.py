import pandas as pd
import json
import folium as fl
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from streamlit_folium import st_folium
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import ast
import http.client
from urllib.request import urlopen
import numpy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



#******************************************************
# FUNCIONES AUXILIARES
#******************************************************

def show_map():
    #Despliegue de mapa
    st.write("Seleccione en el siguiente mapa dónde desea realizar la predicción:")
    m = fl.Map(location=[40.3912, -3.6584], zoom_start=5, min_zoom =5, max_zoom=5)
    m.add_child(fl.LatLngPopup())
    map = st_folium(m, height=350, width=700, returned_objects=["last_clicked", "bounds"])
    #coordenadas clickeadas en el mapa

    try:
        return [map['last_clicked']['lat'],map['last_clicked']['lng']]
    except:
        return 0

    

def calc_dist(x ,df, coord):
    return ((float(df.loc[x]["lon"])-coord[1])**2 + (float(df.loc[x]["lat"])-coord[0])**2)**0.5

@st.cache_data
def cargar_datos(start_year):

    año = start_year
    mes = ["01","02","03","04","05","10","11","12"]
    dia = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]

    with open('datos/1992-01-01.json', 'r') as file:
        datosSmall = json.load(file)
        
    for x in range(año, 2022):
        for m in mes:
            for d in dia:
                fecha = str(x)+"-"+m+"-"+d
                print(fecha)
                try:
                    with open('datos/'+str(x)+'/'+fecha+'.json', 'r') as file:
                        datosSmall += json.load(file)
                except:
                    print("no json file in fecha -> "+ fecha)
    print("fin de carga de datos. start json_normalize")
    
    df = pd.json_normalize(datosSmall)
    print("NORMALIZE SUCCESFUL")
    return df

def check_datos(datosDF, dtEstaciones):
    repeat = True
    while(repeat):
        repeat = False

        estacionCercana = dtEstaciones[dtEstaciones['dist'] == dtEstaciones['dist'].min()]
        datosLocalizados = datosDF[datosDF['indicativo']==estacionCercana.iloc[0,1]]
        #comprueba que existan datos históricos de dicha estación
        if (datosLocalizados.empty):
            repeat = True
            dtEstaciones = dtEstaciones.drop(estacionCercana.index)
            print(f"Eliminada estación: {estacionCercana.iloc[0,1]} por falta de datos.")
        #comprueba que existen registros de todas las variables
        else:
            for var in ["tmed","prec","tmin","tmax", "dir", "velmedia", "racha", "sol", "presMax", "presMin","horatmin","horatmax","horaracha","horaPresMax","horaPresMin"]:
                if(datosLocalizados[var].isnull().sum()==datosLocalizados.shape[0]):
                    repeat = True
                    dtEstaciones = dtEstaciones.drop(estacionCercana.index)
                    print(f"Eliminada estación: {estacionCercana.iloc[0,1]} por falta de variable: {var}")
                    break
        #comprueba que haya algún valor de clase = 1
        if(repeat==False):
            notmin = True
            check_for_nan = datosLocalizados['tmin'].isnull()
            for bol,val in zip(check_for_nan, datosLocalizados['tmin']):
                if not(bol):
                    if('-' in val):
                        notmin = False
                        break
            if(notmin):
                repeat = True
                dtEstaciones = dtEstaciones.drop(estacionCercana.index)
                print(f"Eliminada estación: {estacionCercana.iloc[0,1]} por falta de valores de la variable clase.")
           
    return datosLocalizados, estacionCercana, dtEstaciones,datosDF


def calcular_estacionCercana(coord):
    print("CALCULANDO ESTACION")
    
    
    # carga de estaciones
    with open('aemet.json', encoding="utf-8") as file:
        estaciones = json.load(file)
    dtEstaciones = pd.json_normalize(estaciones)
    dtEstaciones['index']=dtEstaciones.index
    # carga de datos
    datosDF = cargar_datos(1992)
    dtEstaciones['dist']=dtEstaciones['index'].apply(calc_dist, args=(dtEstaciones, coord))

    return check_datos(datosDF, dtEstaciones)
    

def reCalcular_estacionCercana(eDF, dDF, aBorrar):
    print("RECALCULANDO ESTACION")
    
    dtEstaciones = eDF
    datosDF = dDF
    print(f"Eliminada estación: {dtEstaciones.iloc[aBorrar,0]}")
    dtEstaciones = dtEstaciones.drop(aBorrar)

    return check_datos(datosDF, dtEstaciones)
      

def generar_helada(x ,df):
    siz = df['helada'].size
    if x < siz-1:
        if not(df.loc[x+1].isnull()['tmin']):
            if df.loc[x+1]['tmin'] <0:
                return 1
            else:
                return 0
        else:
            return 0
    else:
            return 0
    
def to_int(x):
        if x == 'NoValue':
                return numpy.nan
        return float(x)

def discretizar_hora(x):
    if x<6:
        return 0
    elif x<12:
        return 1
    elif x<18:
        return 2
    else:
        return 3

def entrenar_modelo(datosLocalizados):
     
    print("datos localizados")
    print(datosLocalizados)

    #PREPROCESAMIENTO DE DATOS
    datosLocalizados["prec"] = datosLocalizados["prec"].str.replace('Ip', '0.05')
    datosLocalizados["prec"] = datosLocalizados["prec"].str.replace('Acum', 'NoValue')
    datosLocalizados["dir"] = datosLocalizados["dir"].str.replace('99', 'NoValue')
    datosLocalizados["dir"] = datosLocalizados["dir"].str.replace('88', 'NoValue')

    for var in ["tmed","prec","tmin","tmax", "velmedia", "racha", "sol", "presMax","presMin"]:
        datosLocalizados[var] = datosLocalizados[var].str.replace(',', '.')
    for var in ["horatmin","horatmax","horaracha", "horaPresMax","horaPresMin"]:
        datosLocalizados[var] = datosLocalizados[var].str.replace(':', '.')
        datosLocalizados[var] = datosLocalizados[var].str.replace('Varias', 'NoValue')

    for var in ["tmed","prec","tmin","tmax", "dir", "velmedia", "racha", "sol", "presMax", "presMin","horatmin","horatmax","horaracha","horaPresMax","horaPresMin"]:
        print(var)
        datosLocalizados[var] = datosLocalizados[var].apply(to_int)

    dLocalizadosNum = datosLocalizados.drop(columns=['fecha','indicativo','nombre','provincia','altitud'])
    print("INNECESARY COLUMNS DROPPED")

    imputer = KNNImputer(n_neighbors=8)
    imputer = imputer.fit(dLocalizadosNum)
    dSinNan = imputer.transform(dLocalizadosNum)
    print(dSinNan.shape)
    print("creando variable HELADA...")
    dConHelada = pd.DataFrame(dSinNan, columns = dLocalizadosNum.columns)

    for var in ['horatmin', 'horatmax','horaracha','horaPresMax','horaPresMin']:
        dConHelada[var] = dConHelada[var].apply(discretizar_hora)

    dConHelada["helada"]=0
    dConHelada['index']=dConHelada.index
    dConHelada['helada']=dConHelada['index'].apply(generar_helada, args=(dConHelada, ))
    dConHelada = dConHelada.drop(columns=['index'])
    print("creada HELADA")

    X = dConHelada.iloc[:-1,[0,1,2,4,6,7,8,10,11,13]]
    Y = dConHelada.iloc[:-1,15:16]

    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)

    X2 = pd.DataFrame(rescaledX, columns = ["tmed","prec","tmin","tmax", "dir", "velmedia", "racha", "sol", "presMax", "presMin"])
    for var in ["horatmin","horatmax","horaracha","horaPresMax","horaPresMin"]:
        X2[var] = dConHelada[var]

    # #balanceo de datos
    oversample = SMOTE(sampling_strategy=0.4)
    under = RandomUnderSampler(sampling_strategy=0.5)

    X2, Y = oversample.fit_resample(X2, Y)
    X2, Y = under.fit_resample(X2, Y)
    print("HELADA balanceada")
    ExtraTree = ExtraTreesClassifier(max_features= 2, min_samples_leaf=1, n_estimators=350).fit(X2, Y)
    print("modelo entrenado")
    return scaler, ExtraTree, imputer

def get_datosHoy(scaler, idema, imputer):
    
    #llamada a api
    conn = http.client.HTTPSConnection("opendata.aemet.es")
    headers = {
        'cache-control': "no-cache"
        }
    url2 = 'https://opendata.aemet.es/opendata/api/observacion/convencional/datos/estacion/'+idema
    api_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJKYXZpZXJwZXRhMDVAaG90bWFpbC5jb20iLCJqdGkiOiJhMzA2NDQ0Zi1kZTQ2LTRlNTMtOTk2OS05ZjJlZDliYjk2N2QiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTY3Mzk3OTg4MCwidXNlcklkIjoiYTMwNjQ0NGYtZGU0Ni00ZTUzLTk5NjktOWYyZWQ5YmI5NjdkIiwicm9sZSI6IiJ9.quob28wwiF-OHfipWCLWGinhbQiQAS5rLofuDd0mqAU"

    conn.request("GET", url2 + "?api_key="+ api_key, headers=headers)
    res = conn.getresponse()
    data = res.read()
    aux = data.decode("utf-8")
    dictionary = ast.literal_eval(aux)

    response = urlopen(dictionary["datos"])
    json_data = response.read().decode('utf-8', 'replace')
    d = json.loads(json_data)
    respuestaAPI = pd.json_normalize(d)

    datosVar = pd.DataFrame(respuestaAPI, columns = ["fint","ta","prec","tamin","tamax", "dv", "vv", "vmax", "inso", "pres"])

    #TRATAMIENTO DE LOS DATOS
    d = {'tmed':datosVar['ta'].mean(),
         'prec':datosVar['prec'].mean(),
         'tmin':datosVar['tamin'].min(),
         'horatmin':datosVar[datosVar['tamin'] == datosVar['tamin'].min()]['fint'].to_string()[-8:][0:5].replace(':', '.'),
         'tmax':datosVar['tamax'].mean(),
         'horatmax':datosVar[datosVar['tamax'] == datosVar['tamax'].max()]['fint'].to_string()[-8:][0:5].replace(':', '.'),
         'dir':datosVar['dv'].mean(),
         'velmedia':datosVar['vv'].mean(),
         'racha':datosVar['vmax'].max(),
         'horaracha':datosVar[datosVar['vmax'] == datosVar['vmax'].max()]['fint'].to_string()[-8:][0:5].replace(':', '.'),
         'sol':datosVar['inso'].mean(),
         'presMax':datosVar['pres'].max(),
         'horaPresMax':datosVar[datosVar['pres'] == datosVar['pres'].max()]['fint'].to_string()[-8:][0:2],
         'presMin':datosVar['pres'].min(),
         'horaPresMin':datosVar[datosVar['pres'] == datosVar['pres'].min()]['fint'].to_string()[-8:][0:2]}
    datosHoy = pd.DataFrame(data=d, index=[0])
    st.write("Datos a predecir:")
    st.write(datosHoy)

    #NULOS
    bbddSinNull = imputer.transform(datosHoy)
    inputSinNull = pd.DataFrame(bbddSinNull, columns = ["tmed","prec","tmin",'horatmin',"tmax",'horatmax', "dir", "velmedia", "racha",'horaracha', "sol", "presMax",'horaPresMax', "presMin",'horaPresMin'])
    #ESCALADO
    toScale = inputSinNull.iloc[:,[0,1,2,4,6,7,8,10,11,13]]
    inputScaled = scaler.transform(toScale)
    inputScaled = pd.DataFrame(inputScaled, columns = ["tmed","prec","tmin","tmax", "dir", "velmedia", "racha", "sol", "presMax", "presMin"])

    toPredict = inputScaled.copy()
    for var in ["horatmin","horatmax","horaracha","horaPresMax","horaPresMin"]:
        datosHoy[var] = datosHoy[var].apply(to_int)
        datosHoy[var] = datosHoy[var].apply(discretizar_hora)
        toPredict[var] = datosHoy[var]
    
    print("Datos a predecir:")
    print(toPredict)
    return toPredict

def predecir(scaler, model, idema, imputer):
    datosHoy = get_datosHoy(scaler, idema, imputer)
    return model.predict(datosHoy)



def run_again(dfE, dDF, aBorrar):
    datosLocalizados, estacionCercana, dfEstaciones, datosDF = reCalcular_estacionCercana(dfE, dDF, aBorrar)
    print(f"entrenando modelo con: {estacionCercana.iloc[0, 0]}")
    try:
        scaler, ExtraTree, imputer = entrenar_modelo(datosLocalizados)

        try:
            prediccion = predecir(scaler, ExtraTree, estacionCercana.iloc[0, 1], imputer)
            st.write("Predicción realizada sobre estacion:")
            st.write(estacionCercana.drop(columns=['index']))

            st.title('Resultado de la predicción:')
            if(prediccion==0):
                st.subheader("En las próximas 24 horas NO se espera ninguna helada :sun_with_face:")
            else:
                st.subheader("En las próximas 24 horas SÍ se espera una helada :snowflake:")
            print("EJECUCUION EXITOSA")
        except:
            print("Execept en AGAIN ejecucion en PREDECIR")
            run_again(dfEstaciones, datosDF, estacionCercana.index)

    except:
        print("Execept en AGAIN ejecucion en ENTRENAR_MODELO")
        run_again(dfEstaciones, datosDF, estacionCercana.index)

    
    

    
    
# CÓDIGO PRINCIPAL

#get_esatcionYdatos()

# Despliegue de mapa y cálculo de estación mása cercana.
coord = 0
coord = show_map()
if coord==0:
    st.write("(Haga clic en el mapa)")
else:
    st.write("Coordenadas seleccionadas:")
    st.write(coord)
    datosLocalizados, estacionCercana, dfEstaciones, datosDF = calcular_estacionCercana(coord)

    print(f"entrenando modelo con: {estacionCercana.iloc[0, 0]}")

    try:
        scaler, ExtraTree, imputer = entrenar_modelo(datosLocalizados)

        try:
            prediccion = predecir(scaler, ExtraTree, estacionCercana.iloc[0, 1], imputer)
            st.write("Predicción realizada sobre estacion:")
            st.write(estacionCercana.drop(columns=['index']))

            st.title('Resultado de la predicción:')
            if(prediccion==0):
               st.subheader("En las próximas 24 horas NO se espera ninguna helada :sun_with_face:")
            else:
                st.subheader("En las próximas 24 horas SI se espera una helada :snowflake:")
            print("EJECUCUION EXITOSA")
        except:
            print("Execept en PRIMERA ejecucion en PREDECIR")
            run_again(dfEstaciones, datosDF, estacionCercana.index)

    except:
        print("Execept en PRIMERA ejecucion en ENTRENAR_MODELO")
        run_again(dfEstaciones, datosDF, estacionCercana.index)


