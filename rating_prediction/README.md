# How to Run Scripts

I ran all these notebooks in Google Collab Pro (GPU utilization). In theory, they can run on any environment with minor refactoring.  

Please move the associated json file with the reviews + ratings into your 'My Drive'. The rationale for reading the data from your Google Drive was so it was easiable accessible by Google Collab (google support). Ideally, we should throw it in a Github repo and scrape it, but the size of the files are to big. 

If you choose not to do this, please load in the json file however you would like just as long it is converted to a dataframe object.

ex) df = pd.read_json('cell_phones_accessories.json', lines=True)
