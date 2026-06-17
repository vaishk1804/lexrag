"""Tests for chunking strategies."""

import pytest
from src.ingestion.chunker import chunk_document, chunk_documents, token_length


# A realistic contract sample — long enough to produce multiple chunks
SAMPLE_CONTRACT = """
MASTER SERVICE AGREEMENT

This Master Service Agreement ("Agreement") is entered into as of January 1, 2024,
between Acme Corporation ("Client") and Beta Services LLC ("Provider").

ARTICLE 1 - DEFINITIONS
1.1 "Services" means the professional services described in each Statement of Work.
1.2 "Confidential Information" means any non-public information disclosed by either party.
1.3 "Intellectual Property" means all patents, trademarks, copyrights, and trade secrets.

ARTICLE 2 - SERVICES
2.1 Provider shall perform the Services described in each SOW attached hereto.
2.2 Provider shall assign qualified personnel to perform the Services.
2.3 Provider may subcontract portions of the Services with prior written consent of Client.

ARTICLE 3 - PAYMENT
3.1 Client shall pay Provider within 30 days of invoice receipt.
3.2 Late payments accrue interest at 1.5% per month on the outstanding balance.
3.3 Client shall reimburse Provider for pre-approved expenses within 15 days.

ARTICLE 4 - TERMINATION
4.1 Either party may terminate this Agreement with 30 days written notice.
4.2 Client may terminate immediately upon Provider's material breach.
4.3 Upon termination, Client shall pay for all Services rendered through termination date.
4.4 Sections 5, 6, and 7 shall survive any termination of this Agreement.

ARTICLE 5 - INDEMNIFICATION
5.1 Each party shall indemnify, defend, and hold harmless the other party from any
claims, damages, or expenses arising from its own negligence or willful misconduct.
5.2 Provider shall indemnify Client against any third-party intellectual property claims.

ARTICLE 6 - LIMITATION OF LIABILITY
6.1 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL,
OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS AGREEMENT.
6.2 Each party's total liability shall not exceed the fees paid in the 12 months
preceding the claim giving rise to such liability.

ARTICLE 7 - CONFIDENTIALITY
7.1 Each party agrees to maintain the confidentiality of the other party's
Confidential Information and not to disclose it to any third party.
7.2 This obligation survives termination for a period of three years.

ARTICLE 8 - GOVERNING LAW
8.1 This Agreement shall be governed by the laws of the State of Delaware.
8.2 Any disputes shall be resolved through binding arbitration in New York, NY.
""" * 4  # repeat to make it long enough to chunk meaningfully

def test_chunk_produces_output():
  """Basic check - chunking a contract produces atleast one chunk"""
  chunks=chunk_document(SAMPLE_CONTRACT,
                        metadata={"source":"test.txt"},
                        chunk_size=512,)
  assert len(chunks) > 0

def test_smaller_chunk_size_produces_more_chunks():
  chunks_256 = chunk_document(
    SAMPLE_CONTRACT,
    metadata={"source":"test.txt"},
    chunk_size=256,
  )

  chunks_512=chunk_document(
    SAMPLE_CONTRACT,
    metadata={"source":"test.txt"},
    chunk_size=512,
  )

  assert len(chunks_256)>=len(chunks_512)

def test_chunk_metadata_carried_over():
  meta = {"source":"contract_001.txt","contract_type":"MSA"}
  chunks=chunk_document(SAMPLE_CONTRACT,metadata=meta,chunk_size=512)
  for chunk in chunks:
    assert chunk.metadata["source"] == "contract_001.txt"
    assert chunk.metadata["contract_type"] == "MSA"
    assert "chunk_index" in chunk.metadata
    assert "char_length" in chunk.metadata
    assert "token_length" in chunk.metadata

def test_chunk_index_is_sequential():
    chunks = chunk_document(
        SAMPLE_CONTRACT,
        metadata={"source": "test.txt"},
        chunk_size=512,
    )
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_no_empty_chunks():
    chunks = chunk_document(
        SAMPLE_CONTRACT,
        metadata={"source": "test.txt"},
        chunk_size=512,
    )
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0


def test_chunk_documents_batch():
    docs = [
        {"text": SAMPLE_CONTRACT, "metadata": {"source": "doc1.txt"}},
        {"text": SAMPLE_CONTRACT, "metadata": {"source": "doc2.txt"}},
    ]
    chunks = chunk_documents(docs, chunk_size=512)
    sources = {c.metadata["source"] for c in chunks}
    print(sources)
    assert "doc1.txt" in sources
    assert "doc2.txt" in sources


def test_chunk_overlap_config_stored():
    chunks = chunk_document(
        SAMPLE_CONTRACT,
        metadata={"source": "test.txt"},
        chunk_size=512,
        chunk_overlap=50,
    )
    for chunk in chunks:
        assert chunk.metadata["chunk_overlap_config"] == 50
        assert chunk.metadata["chunk_size_config"] == 512


def test_invalid_text_raises():
    """Empty or non-string text should raise ValueError."""
    with pytest.raises(ValueError):
        chunk_document("", metadata={"source": "test.txt"})
    with pytest.raises(ValueError):
        chunk_document(None, metadata={"source": "test.txt"})


def test_invalid_metadata_raises():
    """Non-dict metadata should raise ValueError."""
    with pytest.raises(ValueError):
        chunk_document(SAMPLE_CONTRACT, metadata="not a dict")


def test_chunk_documents_skips_invalid_docs():
    """Invalid docs should be skipped, not crash the pipeline."""
    docs = [
        {"text": SAMPLE_CONTRACT, "metadata": {"source": "good.txt"}},
        {"missing_text": "bad doc"},       # missing text key
        "not a dict at all",               # not a dict
        {"text": SAMPLE_CONTRACT, "metadata": {"source": "also_good.txt"}},
    ]
    chunks = chunk_documents(docs, chunk_size=512)
    sources = {c.metadata["source"] for c in chunks}
    print(sources)
    assert "good.txt" in sources
    assert "also_good.txt" in sources


def test_token_length_is_real_tokens():
    """
    Verify token_length uses real tokenization, not character approximation.
    'Hello' is 1 token, not 5 chars / 4 = 1.25.
    'indemnification' is 4 tokens, not 15 chars / 4 = 3.75.
    """
    assert token_length("Hello") == 1
    assert token_length("indemnification") == 4


def test_short_important_clause_preserved():
    """Short clauses with legal significance should not be filtered out."""
    doc_with_short_clause = SAMPLE_CONTRACT + "\nGoverning Law: Delaware."
    chunks = chunk_document(
        doc_with_short_clause,
        metadata={"source": "test.txt"},
        chunk_size=512,
    )
    all_text = " ".join(c.text for c in chunks)
    assert "Governing Law" in all_text