# overide LiteralE/main_literal.py 
cp main_literal.py LiteralE

# overide LiteralE/model.py 
cp model.py LiteralE

# added semantic-constraints components to LiteralE data folder
cp domain_per_rel.json LiteralE/data
cp range_per_rel.json LiteralE/data
cp entity_types.json LiteralE/data

# add LiteralE/run_missing_triples_prediction.py 
cp run_missing_triples_prediction.py LiteralE

cp LiteralE-models.tar.gz LiteralE && cd LiteralE && tar -xvf LiteralE-models.tar.gz

# back to KHE-triples folder 
cd ..