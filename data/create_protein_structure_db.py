import kagglehub
path = kagglehub.dataset_download("shahir/protein-data-set")

print("Path to dataset files:", path)

from collections import Counter

df = pd.read_csv('data/pdb_data_no_dups.csv').merge(pd.read_csv('data/pdb_data_seq.csv'), how='inner', on='structureId').drop_duplicates(["sequence"]) # ,"classification"
# Drop rows with missing labels
df = df[[type(c) == type('') for c in df.classification.values]]
df = df[[type(c) == type('') for c in df.sequence.values]]
# select proteins
df = df[df.macromoleculeType_x == 'Protein']
df.reset_index()
print(df.shape)
max_length = 300
df = df.loc[df.residueCount_x<300]

cnt = Counter(df.classification)
# select only K most common classes! - was 10 by default
top_classes = 10
# sort classes
sorted_classes = cnt.most_common()[:top_classes]
classes = [c[0] for c in sorted_classes]
counts = [c[1] for c in sorted_classes]
print("at least " + str(counts[-1]) + " instances per class")

# apply to dataframe
print(str(df.shape[0]) + " instances before")
df = df[[c in classes for c in df.classification]]

df = df[['structureId','classification', 'sequence']]
class_nam = df['classification']
unique_classes = sorted(set(class_nam))
class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
print("class_to_idx", class_to_idx)
int_labels = [torch.tensor(class_to_idx[nam]) for nam in class_nam]
#tokenized_seqs = tokenize(sequence, max_len)
data = pd.DataFrame({
    "class" : class_nam,
    "class_ints" : int_labels,
    "sequence" : df['sequence'],
    "structureId" : df['structureId']
})

train_df, test_df = train_test_split(
            data, 
            test_size=0.15, 
            shuffle=True, 
            random_state=42
        )
dev_rel = 0.15 / 0.85
train_df, dev_df = train_test_split(
    data, 
    test_size=dev_rel, 
    shuffle=True, 
    random_state=42
)
train_df["split_group"] = "train"
dev_df["split_group"] = "dev"
test_df["split_group"] = "test"

combined = pd.concat([train_df, dev_df, test_df], ignore_index=True)
combined["class_ints"] = combined["class_ints"].astype(int)
combined.to_csv('data/protein_struct_classification.csv', index = False)