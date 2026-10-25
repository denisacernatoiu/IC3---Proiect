// cod pentru selectia datelor din Google Earth Engine 
// codul trebuie rulat in GEE Code Editor si APOI descarcaT manual

var bucharest = ee.Geometry.Rectangle([25.95, 44.33, 26.25, 44.55]);

var startDate = ee.Date('2024-01-01');
var endDate = ee.Date('2026-10-26');

var band = 'tropospheric_NO2_column_number_density';

var collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
  .filterBounds(bucharest)
  .filterDate(startDate, endDate)
  .select(band);

print('Numar total imagini:', collection.size());

var numDays = endDate.difference(startDate, 'day');
var days = ee.List.sequence(0, numDays.subtract(1));

var dailyTable = ee.FeatureCollection(days.map(function(i) {
  var date = startDate.advance(i, 'day');
  var nextDate = date.advance(1, 'day');
  var dailyImages = collection.filterDate(date, nextDate);
  var imageCount = dailyImages.size();

  // Valoare implicita pentru zile fara imagini
  var value = ee.Algorithms.If(
    imageCount.gt(0),
    dailyImages.mean().reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: bucharest,
      scale: 1113.2,
      maxPixels: 1e8,
      bestEffort: true
    }).get(band, null),
    null
  );

  return ee.Feature(null, {date: date.format('YYYY-MM-dd'), no2_column_mol_m2: value, num_images: imageCount});
}));

print('Primele rezultate zilnice:', dailyTable.limit(10));
print('Numar de zile:', dailyTable.size());

Export.table.toDrive({
  collection: dailyTable,
  description: 'UrbanTwin_S5P_daily_2024_2026',
  fileNamePrefix: 'UrbanTwin_S5P_daily_2024_2026',
  fileFormat: 'CSV'
});