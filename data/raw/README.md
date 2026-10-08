Place the Low Carbon London smart-meter data here (it is NOT committed to git).

Either run `python -m src.data.download`, or download it yourself from
https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
and drop `Partitioned LCL Data.zip` here. Do not unzip it; the preprocessing reads the CSVs
straight out of the zip. Loose CSVs (London Datastore or Kaggle "halfhourly_dataset/block_*.csv"
format) also work. See data/README.md.
