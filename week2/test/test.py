# Copyright 2008 by Bartek Wilczynski.  All rights reserved.
# Revisions copyright 2019 by Victor Lin.
# Adapted from test_Mymodule.py by Jeff Chang.
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Tests for motifs module."""

if hasattr(str, 'memcpy'):
    # Codon (change these to import our stuff)
    from bio_codon import motifs
    from bio_codon.Seq import Seq
else:
    # Python 
    from Bio import motifs
    from Bio.Seq import Seq
    import numpy as np
    import math
    import tempfile
    import unittest
    def test(func):
        return func  # just return the function unchanged

def normalize_whitespace(s):
    """Convert multiple spaces to single spaces per line, only for strings."""
    if not isinstance(s, str):
        return s
    return "\n".join(" ".join(line.split()) for line in s.splitlines())

def assertEqual(a, b, msg=None):
    a_norm = normalize_whitespace(a)
    b_norm = normalize_whitespace(b)
    if a_norm != b_norm:
        raise AssertionError(msg or f"assertEqual failed:\n{a!r} !=\n{b!r}")

def assertTrue(expr, msg=None):
    if not expr:
        raise AssertionError(msg or f"assertTrue failed: {expr!r} is not True")

def assertFalse(expr, msg=None):
    if expr:
        raise AssertionError(msg or f"assertFalse failed: {expr!r} is not False")

def assertRaises(expected_exception, func, *args, **kwargs):
    """
    Asserts that calling func(*args, **kwargs) raises expected_exception.
    Usage: assertRaises(ValueError, int, "abc")
    """
    try:
        func(*args, **kwargs)
    except expected_exception:
        return  # Success, expected exception was raised
    except Exception as e:
        raise AssertionError(
            f"assertRaises failed: Expected {expected_exception}, but got {type(e)}: {e!r}"
        )
    else:
        raise AssertionError(
            f"assertRaises failed: Expected {expected_exception} to be raised, but no exception occurred"
        )

def assertAlmostEqual(a, b, tol=None, places=None, msg=None):
    """
    Asserts that two numbers are approximately equal.
    Use either tol (absolute tolerance) or places (decimal places).
    """
    if places is not None:
        tol = 10 ** -places
    tol = tol if tol is not None else 1e-7
    if abs(a - b) > tol:
        raise AssertionError(msg or f"assertAlmostEqual failed: {a!r} != {b!r} within tolerance {tol}")


def assertIsNone(obj, msg=None):
    if obj is not None:
        raise AssertionError(msg or f"assertIsNone failed: {obj!r} is not None")

@test
def test_format():
    m = motifs.create([Seq("ATATA")])
    m.name = "Foo"
    s1 = format(m, "pfm")
    expected_pfm = """  1.00   0.00   1.00   0.00  1.00
0.00   0.00   0.00   0.00  0.00
0.00   0.00   0.00   0.00  0.00
0.00   1.00   0.00   1.00  0.00
"""
    s2 = format(m, "jaspar")
    expected_jaspar = """>None Foo
A [  1.00   0.00   1.00   0.00   1.00]
C [  0.00   0.00   0.00   0.00   0.00]
G [  0.00   0.00   0.00   0.00   0.00]
T [  0.00   1.00   0.00   1.00   0.00]
"""
    assertEqual(s2, expected_jaspar)
    s3 = format(m, "transfac")
    expected_transfac = """P0      A      C      G      T
01      1      0      0      0      A
02      0      0      0      1      T
03      1      0      0      0      A
04      0      0      0      1      T
05      1      0      0      0      A
XX
//
"""
    assertEqual(s3, expected_transfac)
    assertRaises(ValueError, format, m, "foo_bar")

@test
def test_relative_entropy():
    m = motifs.create([Seq("ATATA"), Seq("ATCTA"), Seq("TTGTA")])
    assertEqual(len(m.alignment), 3)
    assertEqual(m.background, {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
    assertEqual(m.pseudocounts, {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0})
    assertTrue(
        np.allclose(
            m.relative_entropy,
            np.array([1.0817041659455104, 2.0, 0.4150374992788437, 2.0, 2.0]),
        )
    )
    m.background = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    assertTrue(
        np.allclose(
            m.relative_entropy,
            np.array(
                [
                    0.8186697601117167,
                    1.7369655941662063,
                    0.5419780939258206,
                    1.7369655941662063,
                    1.7369655941662063,
                ]
            ),
        )
    )
    m.background = None
    assertEqual(m.background, {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
    pseudocounts = math.sqrt(len(m.alignment))
    m.pseudocounts = {
        letter: m.background[letter] * pseudocounts for letter in "ACGT"
    }
    assertTrue(
        np.allclose(
            m.relative_entropy,
            np.array(
                [
                    0.3532586861097656,
                    0.7170228827697498,
                    0.11859369972847714,
                    0.7170228827697498,
                    0.7170228827697499,
                ]
            ),
        )
    )
    m.background = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    assertTrue(
        np.allclose(
            m.relative_entropy,
            np.array(
                [
                    0.19727984803857979,
                    0.561044044698564,
                    0.20984910512125132,
                    0.561044044698564,
                    0.5610440446985638,
                ]
            ),
        )
    )

@test
def test_reverse_complement():
    """Test if motifs can be reverse-complemented."""
    background = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    pseudocounts = 0.5
    m = motifs.create([Seq("ATATA")])
    m.background = background
    m.pseudocounts = pseudocounts
    received_forward = format(m, "transfac")
    expected_forward = """\
P0      A      C      G      T
01      1      0      0      0      A
02      0      0      0      1      T
03      1      0      0      0      A
04      0      0      0      1      T
05      1      0      0      0      A
XX
//
"""
    assertEqual(received_forward, expected_forward)
    expected_forward_pwm = """\
    0      1      2      3      4
A:   0.50   0.17   0.50   0.17   0.50
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
T:   0.17   0.50   0.17   0.50   0.17
"""
    assertEqual(str(m.pwm), expected_forward_pwm)
    m = m.reverse_complement()
    received_reverse = format(m, "transfac")
    expected_reverse = """\
P0      A      C      G      T
01      0      0      0      1      T
02      1      0      0      0      A
03      0      0      0      1      T
04      1      0      0      0      A
05      0      0      0      1      T
XX
//
"""
    assertEqual(received_reverse, expected_reverse)
    expected_reverse_pwm = """\
    0      1      2      3      4
A:   0.17   0.50   0.17   0.50   0.17
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
T:   0.50   0.17   0.50   0.17   0.50
"""
    assertEqual(str(m.pwm), expected_reverse_pwm)
    # Same but for RNA motif.
    background_rna = {"A": 0.3, "C": 0.2, "G": 0.2, "U": 0.3}
    pseudocounts = 0.5
    m_rna = motifs.create([Seq("AUAUA")], alphabet="ACGU")
    m_rna.background = background_rna
    m_rna.pseudocounts = pseudocounts
    expected_forward_rna_counts = """\
    0      1      2      3      4
A:   1.00   0.00   1.00   0.00   1.00
C:   0.00   0.00   0.00   0.00   0.00
G:   0.00   0.00   0.00   0.00   0.00
U:   0.00   1.00   0.00   1.00   0.00
"""
    assertEqual(str(m_rna.counts), expected_forward_rna_counts)
    expected_forward_rna_pwm = """\
    0      1      2      3      4
A:   0.50   0.17   0.50   0.17   0.50
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
U:   0.17   0.50   0.17   0.50   0.17
"""
    assertEqual(str(m_rna.pwm), expected_forward_rna_pwm)
    expected_reverse_rna_counts = """\
    0      1      2      3      4
A:   0.00   1.00   0.00   1.00   0.00
C:   0.00   0.00   0.00   0.00   0.00
G:   0.00   0.00   0.00   0.00   0.00
U:   1.00   0.00   1.00   0.00   1.00
"""
    assertEqual(
        str(m_rna.reverse_complement().counts), expected_reverse_rna_counts
    )
    expected_reverse_rna_pwm = """\
    0      1      2      3      4
A:   0.17   0.50   0.17   0.50   0.17
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
U:   0.50   0.17   0.50   0.17   0.50
"""
    assertEqual(str(m_rna.reverse_complement().pwm), expected_reverse_rna_pwm)
    # Same thing, but now start with a motif calculated from a count matrix
    m = motifs.create([Seq("ATATA")])
    counts = m.counts
    m = motifs.Motif(counts=counts)
    m.background = background
    m.pseudocounts = pseudocounts
    received_forward = format(m, "transfac")
    assertEqual(received_forward, expected_forward)
    assertEqual(str(m.pwm), expected_forward_pwm)
    m = m.reverse_complement()
    received_reverse = format(m, "transfac")
    assertEqual(received_reverse, expected_reverse)
    assertEqual(str(m.pwm), expected_reverse_pwm)
    # Same, but for RNA count matrix
    m_rna = motifs.create([Seq("AUAUA")], alphabet="ACGU")
    counts = m_rna.counts
    m_rna = motifs.Motif(counts=counts, alphabet="ACGU")
    m_rna.background = background_rna
    m_rna.pseudocounts = pseudocounts
    assertEqual(str(m_rna.counts), expected_forward_rna_counts)
    assertEqual(str(m_rna.pwm), expected_forward_rna_pwm)
    assertEqual(
        str(m_rna.reverse_complement().counts), expected_reverse_rna_counts
    )
    assertEqual(str(m_rna.reverse_complement().pwm), expected_reverse_rna_pwm)


@test
def test_minimal_meme_parser():
    """Parse motifs/minimal_test.meme file."""
    with open("motifs/minimal_test.meme") as stream:
        record = motifs.parse(stream, "minimal")
    assertEqual(record.version, "4")
    assertEqual(record.alphabet, "ACGT")
    assertEqual(len(record.sequences), 0)
    assertEqual(record.command, "")
    assertEqual(len(record), 3)
    motif = record[0]
    assertEqual(motif.name, "KRP")
    assertEqual(record["KRP"], motif)
    assertEqual(motif.num_occurrences, 17)
    assertEqual(motif.length, 19)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["T"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 4.1e-09, places=10)
    assertEqual(motif.alphabet, "ACGT")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "TGTGATCGAGGTCACACTT")
    assertEqual(motif.degenerate_consensus, "TGTGANNNWGNTCACAYWW")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    1.1684297174927525,
                    0.9432809925744818,
                    1.4307101633876265,
                    1.1549413780465179,
                    0.9308256303218774,
                    0.009164393966550805,
                    0.20124190687894253,
                    0.17618542656995528,
                    0.36777933103380855,
                    0.6635834532368525,
                    0.07729943368061855,
                    0.9838293592717438,
                    1.72489868427398,
                    0.8397561713453014,
                    1.72489868427398,
                    0.8455332015343343,
                    0.3106481207768122,
                    0.7382733641762232,
                    0.537435993300495,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "TGATCGA")
    motif = record[1]
    assertEqual(motif.name, "IFXA")
    assertEqual(record["IFXA"], motif)
    assertEqual(motif.num_occurrences, 14)
    assertEqual(motif.length, 18)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["T"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 3.2e-35, places=36)
    assertEqual(motif.alphabet, "ACGT")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "TACTGTATATATATCCAG")
    assertEqual(motif.degenerate_consensus, "TACTGTATATAHAWMCAG")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    0.9632889858595118,
                    1.02677956765017,
                    2.451526420551951,
                    1.7098384161433415,
                    2.2598671267551107,
                    1.7098384161433415,
                    1.02677956765017,
                    1.391583804103081,
                    1.02677956765017,
                    1.1201961888781142,
                    0.27822438781180836,
                    0.36915366971717867,
                    1.7240522753630425,
                    0.3802185945622609,
                    0.790937683007783,
                    2.451526420551951,
                    1.7240522753630425,
                    1.3924085743645374,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "CTGTATA")
    with open("motifs/minimal_test.meme") as stream:
        record = motifs.parse(stream, "minimal")
    motif = record[2]
    assertEqual(motif.name, "IFXA_no_nsites_no_evalue")
    assertEqual(record["IFXA_no_nsites_no_evalue"], motif)
    assertEqual(motif.num_occurrences, 20)
    assertEqual(motif.length, 18)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["T"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 0.0, places=36)
    assertEqual(motif.alphabet, "ACGT")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "TACTGTATATATATCCAG")
    assertEqual(motif.degenerate_consensus, "TACTGTATATAHAWMCAG")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    0.99075309,
                    1.16078104,
                    2.45152642,
                    1.70983842,
                    2.25986713,
                    1.70983842,
                    1.16078104,
                    1.46052586,
                    1.16078104,
                    1.10213019,
                    0.29911041,
                    0.36915367,
                    1.72405228,
                    0.37696488,
                    0.85258086,
                    2.45152642,
                    1.72405228,
                    1.42793329,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "CTGTATA")


    """Test if Bio.motifs can parse MEME output files using RNA."""
    with open("motifs/minimal_test_rna.meme") as stream:
        record = motifs.parse(stream, "minimal")
    assertEqual(record.version, "4")
    assertEqual(record.alphabet, "ACGU")
    assertEqual(len(record.sequences), 0)
    assertEqual(record.command, "")
    assertEqual(len(record), 3)
    motif = record[0]
    assertEqual(motif.name, "KRP_fake_RNA")
    assertEqual(record["KRP_fake_RNA"], motif)
    assertEqual(motif.num_occurrences, 17)
    assertEqual(motif.length, 19)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["U"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 4.1e-09, places=10)
    assertEqual(motif.alphabet, "ACGU")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "UGUGAUCGAGGUCACACUU")
    assertEqual(motif.degenerate_consensus, "UGUGANNNWGNUCACAYWW")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    1.1684297174927525,
                    0.9432809925744818,
                    1.4307101633876265,
                    1.1549413780465179,
                    0.9308256303218774,
                    0.009164393966550805,
                    0.20124190687894253,
                    0.17618542656995528,
                    0.36777933103380855,
                    0.6635834532368525,
                    0.07729943368061855,
                    0.9838293592717438,
                    1.72489868427398,
                    0.8397561713453014,
                    1.72489868427398,
                    0.8455332015343343,
                    0.3106481207768122,
                    0.7382733641762232,
                    0.537435993300495,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "UGAUCGA")
    motif = record[1]
    assertEqual(motif.name, "IFXA_fake_RNA")
    assertEqual(record["IFXA_fake_RNA"], motif)
    assertEqual(motif.num_occurrences, 14)
    assertEqual(motif.length, 18)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["U"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 3.2e-35, places=36)
    assertEqual(motif.alphabet, "ACGU")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "UACUGUAUAUAUAUCCAG")
    assertEqual(motif.degenerate_consensus, "UACUGUAUAUAHAWMCAG")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    0.9632889858595118,
                    1.02677956765017,
                    2.451526420551951,
                    1.7098384161433415,
                    2.2598671267551107,
                    1.7098384161433415,
                    1.02677956765017,
                    1.391583804103081,
                    1.02677956765017,
                    1.1201961888781142,
                    0.27822438781180836,
                    0.36915366971717867,
                    1.7240522753630425,
                    0.3802185945622609,
                    0.790937683007783,
                    2.451526420551951,
                    1.7240522753630425,
                    1.3924085743645374,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "CUGUAUA")

    motif = record[2]
    assertEqual(motif.name, "IFXA_no_nsites_no_evalue_fake_RNA")
    assertEqual(record["IFXA_no_nsites_no_evalue_fake_RNA"], motif)
    assertEqual(motif.num_occurrences, 20)
    assertEqual(motif.length, 18)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["U"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 0.0, places=36)
    assertEqual(motif.alphabet, "ACGU")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "UACUGUAUAUAUAUCCAG")
    assertEqual(motif.degenerate_consensus, "UACUGUAUAUAHAWMCAG")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    0.99075309,
                    1.16078104,
                    2.45152642,
                    1.70983842,
                    2.25986713,
                    1.70983842,
                    1.16078104,
                    1.46052586,
                    1.16078104,
                    1.10213019,
                    0.29911041,
                    0.36915367,
                    1.72405228,
                    0.37696488,
                    0.85258086,
                    2.45152642,
                    1.72405228,
                    1.42793329,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "CUGUAUA")

@test
def test_meme_parser_rna():
    """Test if Bio.motifs can parse MEME output files using RNA."""
    with open("motifs/minimal_test_rna.meme") as stream:
        record = motifs.parse(stream, "minimal")
    assertEqual(record.version, "4")
    assertEqual(record.alphabet, "ACGU")
    assertEqual(len(record.sequences), 0)
    assertEqual(record.command, "")
    assertEqual(len(record), 3)
    motif = record[0]
    assertEqual(motif.name, "KRP_fake_RNA")
    assertEqual(record["KRP_fake_RNA"], motif)
    assertEqual(motif.num_occurrences, 17)
    assertEqual(motif.length, 19)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["U"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 4.1e-09, places=10)
    assertEqual(motif.alphabet, "ACGU")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "UGUGAUCGAGGUCACACUU")
    assertEqual(motif.degenerate_consensus, "UGUGANNNWGNUCACAYWW")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    1.1684297174927525,
                    0.9432809925744818,
                    1.4307101633876265,
                    1.1549413780465179,
                    0.9308256303218774,
                    0.009164393966550805,
                    0.20124190687894253,
                    0.17618542656995528,
                    0.36777933103380855,
                    0.6635834532368525,
                    0.07729943368061855,
                    0.9838293592717438,
                    1.72489868427398,
                    0.8397561713453014,
                    1.72489868427398,
                    0.8455332015343343,
                    0.3106481207768122,
                    0.7382733641762232,
                    0.537435993300495,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "UGAUCGA")
    motif = record[1]
    assertEqual(motif.name, "IFXA_fake_RNA")
    assertEqual(record["IFXA_fake_RNA"], motif)
    assertEqual(motif.num_occurrences, 14)
    assertEqual(motif.length, 18)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["U"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 3.2e-35, places=36)
    assertEqual(motif.alphabet, "ACGU")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "UACUGUAUAUAUAUCCAG")
    assertEqual(motif.degenerate_consensus, "UACUGUAUAUAHAWMCAG")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    0.9632889858595118,
                    1.02677956765017,
                    2.451526420551951,
                    1.7098384161433415,
                    2.2598671267551107,
                    1.7098384161433415,
                    1.02677956765017,
                    1.391583804103081,
                    1.02677956765017,
                    1.1201961888781142,
                    0.27822438781180836,
                    0.36915366971717867,
                    1.7240522753630425,
                    0.3802185945622609,
                    0.790937683007783,
                    2.451526420551951,
                    1.7240522753630425,
                    1.3924085743645374,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "CUGUAUA")

    motif = record[2]
    assertEqual(motif.name, "IFXA_no_nsites_no_evalue_fake_RNA")
    assertEqual(record["IFXA_no_nsites_no_evalue_fake_RNA"], motif)
    assertEqual(motif.num_occurrences, 20)
    assertEqual(motif.length, 18)
    assertAlmostEqual(motif.background["A"], 0.30269730269730266)
    assertAlmostEqual(motif.background["C"], 0.1828171828171828)
    assertAlmostEqual(motif.background["G"], 0.20879120879120877)
    assertAlmostEqual(motif.background["U"], 0.30569430569430567)
    assertAlmostEqual(motif.evalue, 0.0, places=36)
    assertEqual(motif.alphabet, "ACGU")
    assertIsNone(motif.alignment)
    assertEqual(motif.consensus, "UACUGUAUAUAUAUCCAG")
    assertEqual(motif.degenerate_consensus, "UACUGUAUAUAHAWMCAG")
    assertTrue(
        np.allclose(
            motif.relative_entropy,
            np.array(
                [
                    0.99075309,
                    1.16078104,
                    2.45152642,
                    1.70983842,
                    2.25986713,
                    1.70983842,
                    1.16078104,
                    1.46052586,
                    1.16078104,
                    1.10213019,
                    0.29911041,
                    0.36915367,
                    1.72405228,
                    0.37696488,
                    0.85258086,
                    2.45152642,
                    1.72405228,
                    1.42793329,
                ]
            ),
        )
    )
    assertEqual(motif[2:9].consensus, "CUGUAUA")

if __name__ == "__main__":
    # run all functions starting with "test_"
    for name, func in list(globals().items()):
        if callable(func) and name.startswith("test_"):
            print(f"Running {name}...")
            func()
    print("All tests finished.")
