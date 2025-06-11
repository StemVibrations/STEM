import folium
import pandas as pd
from ipywidgets import Checkbox, VBox, Output
from IPython.display import display
from pyproj import Transformer

# ====== 1. Data: RD-coördinaten van de sonderingen ======
data = {
    'Naam': ['GD231254-SW01', 'GD231254-SW02', 'GD231254-SW03'],
    'X_RD': [228270.993, 228284.419, 228292.590],  # voorbeeldcoördinaten in meters
    'Y_RD': [478794.529, 478802.362, 478807.305]
}
df = pd.DataFrame(data)

# ====== 2. Omzetten RD naar GPS (lat/lon) ======
def rd_to_wgs84(x, y):
    transformer = Transformer.from_crs("epsg:28992", "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon

df[['Lat', 'Lon']] = df.apply(lambda row: pd.Series(rd_to_wgs84(row['X_RD'], row['Y_RD'])), axis=1)

# ====== 3. Kaart initialiseren ======
map_center = [df['Lat'].mean(), df['Lon'].mean()]
m = folium.Map(location=map_center, zoom_start=17)

# ====== 4. Markers voorbereiden ======
markers = {}
for idx, row in df.iterrows():
    marker = folium.Marker(
        location=[row['Lat'], row['Lon']],
        popup=row['Naam'],
        icon=folium.Icon(color="blue", icon="info-sign")
    )
    markers[row['Naam']] = marker

# ====== 5. Voeg markers toe aan kaart ======
for marker in markers.values():
    marker.add_to(m)

# ====== Extra: Marker voor station Holten toevoegen ======
station_lat = 52.284120442802376
station_lon = 6.421428606616156

station_marker = folium.Marker(
    location=[station_lat, station_lon],
    popup="Station Holten",
    icon=folium.Icon(color="red", icon="train", prefix='fa')
)
station_marker.add_to(m)

# ====== Extra: Marker voor meetlocatie 024 KM 21,605 toevoegen met RD-coördinaten ======
x_rd = 228277.07
y_rd = 478800.59

# RD naar WGS84 omzetten
meetloc_lat, meetloc_lon = rd_to_wgs84(x_rd, y_rd)

meetloc_marker = folium.Marker(
    location=[meetloc_lat, meetloc_lon],
    popup="Meetlocatie 024 KM 21,605",
    icon=folium.Icon(color="green", icon="star")
)
meetloc_marker.add_to(m)

# ====== Legenda toevoegen ======
legend_html = """
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 200px; height: 110px; 
    background-color: white; 
    border:2px solid grey; z-index:9999; font-size:14px;
    padding: 10px;
">
<b>Legenda</b><br>
<span style="color:blue;">&#9679;</span> CPT<br>
<span style="color:red;">&#9679;</span> Station Holten<br>
<span style="color:green;">&#9679;</span> Measurement location </div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# ====== 6. Kaart opslaan en openen in browser ======
m.save("sonderingen_kaart.html")

import webbrowser
webbrowser.open("sonderingen_kaart.html")


