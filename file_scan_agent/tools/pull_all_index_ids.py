import os
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import  Optional

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def pull_all_index_ids(
    index_name: str,
    name_space: Optional[str] = None,
) -> list[str]:
    """
    Retrieves every ID from a Pinecone index.
    
    Args:
        index_name:     The name for your index (e.g. "abc123-us-west1-gcp").
        name_space:      (Optional) the namespace to target. If omitted, default namespace is used.

    Returns:
        A list of ids in the index
    """
    # Initialize client and target the index
    index = pc_client.Index(index_name)
    
    ids = index.list(namespace=name_space)
    # Gets list of lists of IDs by page
    ids = list(ids)
    # Flatten the list of lists
    ids = [item for sublist in ids for item in sublist]

    return ids

def pull_all_index_prefixes(
    index_name: str,
    name_space: Optional[str] = None,
) -> list[str]:
    """
    Retrieves every ID prefix from a Pinecone index.

    Args:
        index_name:     The name for your index (e.g. "abc123-us-west1-gcp").
        name_space:      (Optional) the namespace to target. If omitted, default namespace is used.

    Returns:
        A set of the prefixes of the IDs in the index.
    """
    # All ids
    ids = pull_all_index_ids(index_name, name_space)

    prefixes: set[str] = set()
    for s in ids:
        if '-' in s:
            prefix, _idnum = s.rsplit('-', 1)
            prefixes.add(prefix)
        else:
            # Optionally handle malformed entries however you like:
            prefixes.add(s)
    return prefixes