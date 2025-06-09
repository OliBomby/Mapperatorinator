#!/usr/bin/env python3
# https://community.wandb.ai/t/using-the-python-api-to-delete-models-with-no-tag-minimal/1498?u=turian
# "Rather than using api.artifact_versions, it uses the versions
# method on artifact_collection."

import wandb
from tqdm.auto import tqdm

# dry_run = True
dry_run = False
api = wandb.Api()
project = api.project("osuT5", entity="mappingtools")

for artifact_type in project.artifacts_types():
    if artifact_type.type != "model":
        continue
    collection_artifacts = []
    collections = []
    for artifact_collection in tqdm(artifact_type.collections()):
        artifacts = list(artifact_collection.artifacts())
        for artifact in artifacts:
            if artifact.state != "DELETED":
                collection_artifacts.append((artifact_collection, artifact))
        if len(artifacts) == 0:
            collections.append(artifact_collection)

    for (artifact_collection, artifact) in tqdm(collection_artifacts):
        if len(artifact.aliases) > 0:
            # print out the name of the one we are keeping
            print(f"KEEPING {artifact.name} {artifact.aliases}")
        else:
            if not dry_run:
                artifact.delete()
            else:
                print("")
                print(f"should delete {artifact.name}")
    for artifact_collection in tqdm(collections):
        if not dry_run:
            artifact_collection.delete()
        else:
            print("")
            print(f"should delete {artifact_collection.name}")