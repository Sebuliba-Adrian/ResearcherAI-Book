"""
Automated tests for code examples in data-foundations.md
Runs in CI/CD before deployment
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVectorDatabases:
    """Test vector database examples from Chapter 3, Part 1"""

    def test_embedding_creation(self):
        """Test: Creating embeddings with SentenceTransformers"""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [
            "Neural networks for image classification",
            "Deep learning in computer vision",
            "Convolutional networks for image recognition"
        ]

        embeddings = model.encode(texts)
        assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"

    def test_cosine_similarity(self):
        """Test: Cosine similarity calculations"""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        def cosine_similarity(vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        model = SentenceTransformer('all-MiniLM-L6-v2')

        vec1 = model.encode("neural networks")
        vec2 = model.encode("deep learning")
        vec3 = model.encode("banana recipe")

        sim_related = cosine_similarity(vec1, vec2)
        sim_unrelated = cosine_similarity(vec1, vec3)

        # Related terms should be more similar than unrelated
        assert sim_related > sim_unrelated, \
            f"Expected neural networks & deep learning ({sim_related:.3f}) > neural networks & banana ({sim_unrelated:.3f})"

    def test_faiss_vector_store(self):
        """Test: FAISS vector store implementation"""
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from typing import List

        class FAISSVectorStore:
            def __init__(self, dimension: int = 384):
                self.dimension = dimension
                self.index = faiss.IndexFlatL2(dimension)
                self.documents = []

            def add_documents(self, texts: List[str], embeddings: np.ndarray):
                embeddings_f32 = embeddings.astype('float32')
                self.index.add(embeddings_f32)
                self.documents.extend(texts)

            def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
                query_f32 = query_embedding.astype('float32').reshape(1, -1)
                distances, indices = self.index.search(query_f32, top_k)
                similarities = 1 / (1 + distances[0])
                return [
                    (self.documents[idx], float(sim))
                    for idx, sim in zip(indices[0], similarities)
                    if idx < len(self.documents)
                ]

        # Test with papers
        model = SentenceTransformer('all-MiniLM-L6-v2')
        vector_store = FAISSVectorStore(dimension=384)

        papers = [
            "Attention is all you need - transformer architecture",
            "BERT: Pre-training of deep bidirectional transformers",
            "ResNet: Deep residual learning for image recognition",
        ]

        embeddings = model.encode(papers)
        vector_store.add_documents(papers, embeddings)

        # Search for NLP papers
        query = "transformer models for NLP"
        query_emb = model.encode(query)
        results = vector_store.search(query_emb, top_k=2)

        # Verify transformer paper is in top results
        top_docs = [doc for doc, _ in results]
        assert any("transformer" in doc.lower() or "BERT" in doc for doc in top_docs), \
            f"Expected transformer-related paper in results: {top_docs}"

        # Verify we don't get CV paper as top result
        assert "ResNet" not in results[0][0], \
            "Computer vision paper should not be top result for NLP query"


class TestHandsOnTutorial:
    """Test the 8-step hands-on tutorial"""

    @pytest.fixture
    def sample_data(self):
        """Fixture: Sample CSV data for testing"""
        import pandas as pd
        from io import StringIO

        papers_csv = """domain,title,year,abstract
NLP,Attention Is All You Need,2017,Transformer architecture
NLP,BERT,2018,Bidirectional encoder
CV,ResNet,2015,Residual learning"""

        authors_csv = """name,affiliation,domain
Ashish Vaswani,Google Brain,NLP
Jacob Devlin,Google AI,NLP
Kaiming He,Facebook AI,CV"""

        citations_csv = """citing_paper,cited_paper,citation_type
BERT,Attention Is All You Need,builds_on"""

        concepts_csv = """paper,concept,importance
Attention Is All You Need,self-attention,high
BERT,bidirectional,high"""

        return {
            'papers': pd.read_csv(StringIO(papers_csv)).fillna(''),
            'authors': pd.read_csv(StringIO(authors_csv)).fillna(''),
            'citations': pd.read_csv(StringIO(citations_csv)).fillna(''),
            'concepts': pd.read_csv(StringIO(concepts_csv)).fillna('')
        }

    def test_transform_function(self, sample_data):
        """Test: SPARQL CONSTRUCT transform function"""
        import re
        from rdflib import Graph, Literal
        from rdflib.plugins.sparql.processor import prepareQuery
        import pandas as pd

        def transform(df: pd.DataFrame, construct_query: str, first: bool = False) -> Graph:
            query_graph = Graph()
            result_graph = Graph()
            query = prepareQuery(construct_query)

            invalid_pattern = re.compile(r"[^\w_]+")
            headers = dict((k, invalid_pattern.sub("_", k)) for k in df.columns)

            for _, row in df.iterrows():
                binding = dict((headers[k], Literal(row[k]))
                              for k in df.columns if len(str(row[k])) > 0)
                results = query_graph.query(query, initBindings=binding)
                for triple in results:
                    result_graph.add(triple)
                if first:
                    break

            return result_graph

        # Test with papers
        construct_query = """
PREFIX research: <http://example.org/research#>
CONSTRUCT {
    ?paper a research:Paper .
    ?paper research:hasTitle ?title .
}
WHERE {
    BIND(IRI(CONCAT("http://data.example.org/paper/",
                    REPLACE(?title, " ", "_"))) AS ?paper)
}
"""

        result = transform(sample_data['papers'], construct_query)

        # Should create 2 triples per paper (type + title)
        assert len(result) >= 6, f"Expected at least 6 triples, got {len(result)}"

    def test_knowledge_graph_construction(self, sample_data):
        """Test: Complete knowledge graph construction"""
        import re
        from rdflib import Graph, Literal
        from rdflib.plugins.sparql.processor import prepareQuery
        import pandas as pd

        def transform(df: pd.DataFrame, construct_query: str, first: bool = False) -> Graph:
            query_graph = Graph()
            result_graph = Graph()
            query = prepareQuery(construct_query)

            invalid_pattern = re.compile(r"[^\w_]+")
            headers = dict((k, invalid_pattern.sub("_", k)) for k in df.columns)

            for _, row in df.iterrows():
                binding = dict((headers[k], Literal(row[k]))
                              for k in df.columns if len(str(row[k])) > 0)
                results = query_graph.query(query, initBindings=binding)
                for triple in results:
                    result_graph.add(triple)
                if first:
                    break

            return result_graph

        # Build knowledge graph
        kg = Graph()

        # Add papers
        construct_papers = """
PREFIX research: <http://example.org/research#>
CONSTRUCT {
    ?paper a research:Paper .
    ?paper research:hasTitle ?title .
    ?paper research:publishedYear ?year .
}
WHERE {
    BIND(IRI(CONCAT("http://data.example.org/paper/",
                    REPLACE(?title, " ", "_"))) AS ?paper)
}
"""
        kg += transform(sample_data['papers'], construct_papers)

        # Add citations
        construct_citations = """
PREFIX research: <http://example.org/research#>
CONSTRUCT {
    ?citing research:cites ?cited .
}
WHERE {
    BIND(IRI(CONCAT("http://data.example.org/paper/",
                    REPLACE(?citing_paper, " ", "_"))) AS ?citing)
    BIND(IRI(CONCAT("http://data.example.org/paper/",
                    REPLACE(?cited_paper, " ", "_"))) AS ?cited)
}
"""
        kg += transform(sample_data['citations'], construct_citations)

        # Verify graph has reasonable number of triples
        assert len(kg) >= 9, f"Expected at least 9 triples, got {len(kg)}"

        # Verify we can query it
        query = """
PREFIX research: <http://example.org/research#>
SELECT ?title
WHERE {
    ?paper a research:Paper .
    ?paper research:hasTitle ?title .
}
"""
        results = list(kg.query(query))
        assert len(results) == 3, f"Expected 3 papers, got {len(results)}"

    def test_sparql_queries(self, sample_data):
        """Test: SPARQL SELECT queries work correctly"""
        import re
        from rdflib import Graph, Literal
        from rdflib.plugins.sparql.processor import prepareQuery
        import pandas as pd

        def transform(df: pd.DataFrame, construct_query: str, first: bool = False) -> Graph:
            query_graph = Graph()
            result_graph = Graph()
            query = prepareQuery(construct_query)

            invalid_pattern = re.compile(r"[^\w_]+")
            headers = dict((k, invalid_pattern.sub("_", k)) for k in df.columns)

            for _, row in df.iterrows():
                binding = dict((headers[k], Literal(row[k]))
                              for k in df.columns if len(str(row[k])) > 0)
                results = query_graph.query(query, initBindings=binding)
                for triple in results:
                    result_graph.add(triple)
                if first:
                    break

            return result_graph

        # Build simple graph
        kg = Graph()

        construct_query = """
PREFIX research: <http://example.org/research#>
CONSTRUCT {
    ?paper a research:Paper .
    ?paper research:hasTitle ?title .
    ?paper research:belongsToDomain ?domainIRI .
}
WHERE {
    BIND(IRI(CONCAT("http://data.example.org/paper/",
                    REPLACE(?title, " ", "_"))) AS ?paper)
    BIND(IRI(CONCAT("http://data.example.org/domain/",
                    ?domain)) AS ?domainIRI)
}
"""
        kg += transform(sample_data['papers'], construct_query)

        # Query for NLP papers
        query_nlp = """
PREFIX research: <http://example.org/research#>
SELECT ?title
WHERE {
    ?paper research:belongsToDomain <http://data.example.org/domain/NLP> .
    ?paper research:hasTitle ?title .
}
"""
        results = list(kg.query(query_nlp))
        assert len(results) == 2, f"Expected 2 NLP papers, got {len(results)}"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetworkX visualization may have issues on Windows"
)
class TestVisualization:
    """Test visualization code"""

    def test_rdf_to_networkx_conversion(self):
        """Test: RDF to NetworkX conversion"""
        import networkx as nx
        from rdflib import Graph, URIRef, Namespace, Literal

        def rdf_to_nx(rdf_graph: Graph) -> nx.DiGraph:
            G = nx.DiGraph()
            for s, p, o in rdf_graph:
                subject = str(s).split('/')[-1]
                predicate = str(p).split('#')[-1]
                obj = str(o).split('/')[-1] if isinstance(o, URIRef) else str(o)
                G.add_edge(subject, obj, label=predicate)
            return G

        # Create simple RDF graph
        rdf_g = Graph()
        ns = Namespace("http://example.org/")

        paper = ns.Paper1
        author = ns.Author1
        rdf_g.add((paper, ns.hasTitle, Literal("Test Paper")))
        rdf_g.add((paper, ns.authoredBy, author))

        # Convert
        nx_g = rdf_to_nx(rdf_g)

        assert nx_g.number_of_nodes() > 0, "NetworkX graph should have nodes"
        assert nx_g.number_of_edges() > 0, "NetworkX graph should have edges"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
