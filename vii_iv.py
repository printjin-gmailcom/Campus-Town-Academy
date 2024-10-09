! pip install folium

import folium

seoul_station = [37.555946, 126.972317]
m = folium.Map(location = seoul_station, zoom_start = 13)

folium.Marker(seoul_station, tooltip = 'seoul', popup = 'station').add_to(m)
m

m.save('./지도샘플.html')

