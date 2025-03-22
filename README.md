Read me file
Step to execute Medical Insurance Fraud Detection using Big Data Analytics system.
Import latest data from below source:
https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers 
https://www.cms.gov/priorities/key-initiatives/open-payments/data/dataset-downloads
https://oig.hhs.gov/exclusions/exclusions_list.asp

MongoDB:
Import the data into MongoDB Using the queries that are in MongoDB Queries.PDF file
And Export the data into databricks

Databricks:
Create an account in Databricks community edition.
Create a sample cluster to with any name.
Import Collecting Data and Integration.ipynb file and related datasets into DBFS run the File.
Following are the datasets:
•	FinalProject.Prescriber
•	FinalProject.Prescriber_Details
•	FinalProject.Prescriber_Drug_Cost_Supplies_Clm
•	FinalProject_Prescriber_Brand-1
•	UPDATED
Note you can see data pipeline create in MongoDB in “After creating mongodb datapline datasets” folder
Download the Integrated dataset “medical_insurance_Dataset” that is create in above file And import the dataset into DBFS file and run the Feature Engineering and ML Models.ipynb file




