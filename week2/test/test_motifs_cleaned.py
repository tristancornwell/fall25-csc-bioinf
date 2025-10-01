# Copyright 2008 by Bartek Wilczynski.  All rights reserved.
# Revisions copyright 2019 by Victor Lin.
# Adapted from test_Mymodule.py by Jeff Chang.
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Tests for motifs module."""

import math
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    from Bio import MissingExternalDependencyError

    raise MissingExternalDependencyError(
        "Install numpy if you want to use Bio.motifs."
    ) from None

from Bio import motifs
from Bio.Seq import Seq


# Copyright 2008 by Bartek Wilczynski.  All rights reserved.
# Revisions copyright 2019 by Victor Lin.
# Adapted from test_Mymodule.py by Jeff Chang.
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Tests for motifs module."""

import math
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    from Bio import MissingExternalDependencyError

    raise MissingExternalDependencyError(
        "Install numpy if you want to use Bio.motifs."
    ) from None

from Bio import motifs
from Bio.Seq import Seq


class TestBasic(unittest.TestCase):

    """Basic motif tests."""

    def test_format(self):
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
        self.assertEqual(s2, expected_jaspar)
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
        self.assertEqual(s3, expected_transfac)
        self.assertRaises(ValueError, format, m, "foo_bar")

    def test_relative_entropy(self):
        m = motifs.create([Seq("ATATA"), Seq("ATCTA"), Seq("TTGTA")])
        self.assertEqual(len(m.alignment), 3)
        self.assertEqual(m.background, {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
        self.assertEqual(m.pseudocounts, {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0})
        self.assertTrue(
            np.allclose(
                m.relative_entropy,
                np.array([1.0817041659455104, 2.0, 0.4150374992788437, 2.0, 2.0]),
            )
        )
        m.background = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
        self.assertTrue(
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
        self.assertEqual(m.background, {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
        pseudocounts = math.sqrt(len(m.alignment))
        m.pseudocounts = {
            letter: m.background[letter] * pseudocounts for letter in "ACGT"
        }
        self.assertTrue(
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
        self.assertTrue(
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

    def test_reverse_complement(self):
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
        self.assertEqual(received_forward, expected_forward)
        expected_forward_pwm = """\
        0      1      2      3      4
A:   0.50   0.17   0.50   0.17   0.50
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
T:   0.17   0.50   0.17   0.50   0.17
"""
        self.assertEqual(str(m.pwm), expected_forward_pwm)
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
        self.assertEqual(received_reverse, expected_reverse)
        expected_reverse_pwm = """\
        0      1      2      3      4
A:   0.17   0.50   0.17   0.50   0.17
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
T:   0.50   0.17   0.50   0.17   0.50
"""
        self.assertEqual(str(m.pwm), expected_reverse_pwm)
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
        self.assertEqual(str(m_rna.counts), expected_forward_rna_counts)
        expected_forward_rna_pwm = """\
        0      1      2      3      4
A:   0.50   0.17   0.50   0.17   0.50
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
U:   0.17   0.50   0.17   0.50   0.17
"""
        self.assertEqual(str(m_rna.pwm), expected_forward_rna_pwm)
        expected_reverse_rna_counts = """\
        0      1      2      3      4
A:   0.00   1.00   0.00   1.00   0.00
C:   0.00   0.00   0.00   0.00   0.00
G:   0.00   0.00   0.00   0.00   0.00
U:   1.00   0.00   1.00   0.00   1.00
"""
        self.assertEqual(
            str(m_rna.reverse_complement().counts), expected_reverse_rna_counts
        )
        expected_reverse_rna_pwm = """\
        0      1      2      3      4
A:   0.17   0.50   0.17   0.50   0.17
C:   0.17   0.17   0.17   0.17   0.17
G:   0.17   0.17   0.17   0.17   0.17
U:   0.50   0.17   0.50   0.17   0.50
"""
        self.assertEqual(str(m_rna.reverse_complement().pwm), expected_reverse_rna_pwm)
        # Same thing, but now start with a motif calculated from a count matrix
        m = motifs.create([Seq("ATATA")])
        counts = m.counts
        m = motifs.Motif(counts=counts)
        m.background = background
        m.pseudocounts = pseudocounts
        received_forward = format(m, "transfac")
        self.assertEqual(received_forward, expected_forward)
        self.assertEqual(str(m.pwm), expected_forward_pwm)
        m = m.reverse_complement()
        received_reverse = format(m, "transfac")
        self.assertEqual(received_reverse, expected_reverse)
        self.assertEqual(str(m.pwm), expected_reverse_pwm)
        # Same, but for RNA count matrix
        m_rna = motifs.create([Seq("AUAUA")], alphabet="ACGU")
        counts = m_rna.counts
        m_rna = motifs.Motif(counts=counts, alphabet="ACGU")
        m_rna.background = background_rna
        m_rna.pseudocounts = pseudocounts
        self.assertEqual(str(m_rna.counts), expected_forward_rna_counts)
        self.assertEqual(str(m_rna.pwm), expected_forward_rna_pwm)
        self.assertEqual(
            str(m_rna.reverse_complement().counts), expected_reverse_rna_counts
        )
        self.assertEqual(str(m_rna.reverse_complement().pwm), expected_reverse_rna_pwm)


class TestAlignAce(unittest.TestCase):

    """Testing parsing Cluster-Buster output files."""

    def test_clusterbuster_parsing_and_output(self):
        """Test if Bio.motifs can parse and output Cluster-Buster PFM files."""
        with open("motifs/clusterbuster.pfm") as stream:
            record = motifs.parse(stream, "clusterbuster")
            self.assertEqual(len(record), 3)
            motif = record[0]
            self.assertEqual(motif.name, "MA0004.1")
            self.assertEqual(motif.alphabet, "GATC")
            self.assertEqual(motif.consensus, "CACGTG")
            self.assertEqual(motif.degenerate_consensus, "CACGTG")
            self.assertTrue(
                np.allclose(
                    motif.relative_entropy,
                    np.array(
                        [1.278071905112638, 1.7136030428840439, 2.0, 2.0, 2.0, 2.0]
                    ),
                )
            )
            self.assertEqual(motif[1:-2].consensus, "ACG")
            self.assertEqual(motif.length, 6)
            self.assertIsNone(motif.weight)
            self.assertIsNone(motif.gap)
            self.assertAlmostEqual(motif.counts["G", 0], 0.0)
            self.assertAlmostEqual(motif.counts["G", 1], 1.0)
            self.assertAlmostEqual(motif.counts["G", 2], 0.0)
            self.assertAlmostEqual(motif.counts["G", 3], 20.0)
            self.assertAlmostEqual(motif.counts["G", 4], 0.0)
            self.assertAlmostEqual(motif.counts["G", 5], 20.0)
            self.assertAlmostEqual(motif.counts["A", 0], 4.0)
            self.assertAlmostEqual(motif.counts["A", 1], 19.0)
            self.assertAlmostEqual(motif.counts["A", 2], 0.0)
            self.assertAlmostEqual(motif.counts["A", 3], 0.0)
            self.assertAlmostEqual(motif.counts["A", 4], 0.0)
            self.assertAlmostEqual(motif.counts["A", 5], 0.0)
            self.assertAlmostEqual(motif.counts["T", 0], 0.0)
            self.assertAlmostEqual(motif.counts["T", 1], 0.0)
            self.assertAlmostEqual(motif.counts["T", 2], 0.0)
            self.assertAlmostEqual(motif.counts["T", 3], 0.0)
            self.assertAlmostEqual(motif.counts["T", 4], 20.0)
            self.assertAlmostEqual(motif.counts["T", 5], 0.0)
            self.assertAlmostEqual(motif.counts["C", 0], 16.0)
            self.assertAlmostEqual(motif.counts["C", 1], 0.0)
            self.assertAlmostEqual(motif.counts["C", 2], 20.0)
            self.assertAlmostEqual(motif.counts["C", 3], 0.0)
            self.assertAlmostEqual(motif.counts["C", 4], 0.0)
            self.assertAlmostEqual(motif.counts["C", 5], 0.0)
            motif = record[1]
            self.assertEqual(motif.name, "MA0006.1")
            self.assertEqual(motif.alphabet, "GATC")
            self.assertEqual(motif.consensus, "TGCGTG")
            self.assertEqual(motif.degenerate_consensus, "YGCGTG")
            self.assertTrue(
                np.allclose(
                    motif.relative_entropy,
                    np.array(
                        [
                            0.28206397041108283,
                            1.7501177071668148,
                            1.7501177071668148,
                            1.7501177071668148,
                            2.0,
                            2.0,
                        ]
                    ),
                )
            )
            self.assertEqual(motif[1:-2].consensus, "GCG")
            self.assertEqual(motif.length, 6)
            self.assertIsNone(motif.weight)
            self.assertIsNone(motif.gap)
            self.assertAlmostEqual(motif.counts["G", 0], 2.0)
            self.assertAlmostEqual(motif.counts["G", 1], 23.0)
            self.assertAlmostEqual(motif.counts["G", 2], 0.0)
            self.assertAlmostEqual(motif.counts["G", 3], 23.0)
            self.assertAlmostEqual(motif.counts["G", 4], 0.0)
            self.assertAlmostEqual(motif.counts["G", 5], 24.0)
            self.assertAlmostEqual(motif.counts["A", 0], 3.0)
            self.assertAlmostEqual(motif.counts["A", 1], 0.0)
            self.assertAlmostEqual(motif.counts["A", 2], 0.0)
            self.assertAlmostEqual(motif.counts["A", 3], 0.0)
            self.assertAlmostEqual(motif.counts["A", 4], 0.0)
            self.assertAlmostEqual(motif.counts["A", 5], 0.0)
            self.assertAlmostEqual(motif.counts["T", 0], 11.0)
            self.assertAlmostEqual(motif.counts["T", 1], 1.0)
            self.assertAlmostEqual(motif.counts["T", 2], 1.0)
            self.assertAlmostEqual(motif.counts["T", 3], 1.0)
            self.assertAlmostEqual(motif.counts["T", 4], 24.0)
            self.assertAlmostEqual(motif.counts["T", 5], 0.0)
            self.assertAlmostEqual(motif.counts["C", 0], 8.0)
            self.assertAlmostEqual(motif.counts["C", 1], 0.0)
            self.assertAlmostEqual(motif.counts["C", 2], 23.0)
            self.assertAlmostEqual(motif.counts["C", 3], 0.0)
            self.assertAlmostEqual(motif.counts["C", 4], 0.0)
            self.assertAlmostEqual(motif.counts["C", 5], 0.0)
            motif = record[2]
            self.assertEqual(motif.name, "MA0008.1")
            self.assertEqual(motif.alphabet, "GATC")
            self.assertEqual(motif.consensus, "CAATTATT")
            self.assertEqual(motif.degenerate_consensus, "CAATTATT")
            self.assertTrue(
                np.allclose(
                    motif.relative_entropy,
                    np.array(
                        [
                            0.2549535827226545,
                            1.2358859454459725,
                            2.0,
                            2.0,
                            1.278071905112638,
                            1.7577078109175852,
                            1.7577078109175852,
                            1.5978208097977271,
                        ]
                    ),
                )
            )
            self.assertEqual(motif[1:-2].consensus, "AATTA")
            self.assertEqual(motif.length, 8)
            self.assertEqual(motif.weight, 3.0)
            self.assertEqual(motif.gap, 10.0)
            self.assertAlmostEqual(motif.counts["G", 0], 4.0)
            self.assertAlmostEqual(motif.counts["G", 1], 0.0)
            self.assertAlmostEqual(motif.counts["G", 2], 0.0)
            self.assertAlmostEqual(motif.counts["G", 3], 0.0)
            self.assertAlmostEqual(motif.counts["G", 4], 0.0)
            self.assertAlmostEqual(motif.counts["G", 5], 1.0)
            self.assertAlmostEqual(motif.counts["G", 6], 0.0)
            self.assertAlmostEqual(motif.counts["G", 7], 2.0)
            self.assertAlmostEqual(motif.counts["A", 0], 3.0)
            self.assertAlmostEqual(motif.counts["A", 1], 21.0)
            self.assertAlmostEqual(motif.counts["A", 2], 25.0)
            self.assertAlmostEqual(motif.counts["A", 3], 0.0)
            self.assertAlmostEqual(motif.counts["A", 4], 0.0)
            self.assertAlmostEqual(motif.counts["A", 5], 24.0)
            self.assertAlmostEqual(motif.counts["A", 6], 1.0)
            self.assertAlmostEqual(motif.counts["A", 7], 0.0)
            self.assertAlmostEqual(motif.counts["T", 0], 5.0)
            self.assertAlmostEqual(motif.counts["T", 1], 3.0)
            self.assertAlmostEqual(motif.counts["T", 2], 0.0)
            self.assertAlmostEqual(motif.counts["T", 3], 25.0)
            self.assertAlmostEqual(motif.counts["T", 4], 20.0)
            self.assertAlmostEqual(motif.counts["T", 5], 0.0)
            self.assertAlmostEqual(motif.counts["T", 6], 24.0)
            self.assertAlmostEqual(motif.counts["T", 7], 23.0)
            self.assertAlmostEqual(motif.counts["C", 0], 13.0)
            self.assertAlmostEqual(motif.counts["C", 1], 1.0)
            self.assertAlmostEqual(motif.counts["C", 2], 0.0)
            self.assertAlmostEqual(motif.counts["C", 3], 0.0)
            self.assertAlmostEqual(motif.counts["C", 4], 5.0)
            self.assertAlmostEqual(motif.counts["C", 5], 0.0)
            self.assertAlmostEqual(motif.counts["C", 6], 0.0)
            self.assertAlmostEqual(motif.counts["C", 7], 0.0)
            stream.seek(0)
            self.assertEqual(
                motifs.write(record, "clusterbuster").split(),
                stream.read().split(),
            )
            stream.seek(0)
            self.assertEqual(
                motifs.write(record, "clusterbuster", precision=2).split("\n"),
                [
                    (
                        line
                        if (line.startswith(">") or line.startswith("#"))
                        else "\t".join([f"{x}.00" for x in line.split()])
                    )
                    for line in stream.read().split("\n")
                ],
            )


class TestXMS(unittest.TestCase):

    """Testing parsing xms output files."""

    def test_xms_parsing(self):
        """Test if Bio.motifs can parse and output xms PFM files."""
        with open("motifs/abdb.xms") as stream:
            record = motifs.parse(stream, "xms")
        self.assertEqual(len(record), 1)
        motif = record[0]
        self.assertEqual(motif.name, "Abd-B")
        self.assertEqual(motif.length, 14)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.333333333)
        self.assertAlmostEqual(motif.counts["G", 1], 0.379310345)
        self.assertAlmostEqual(motif.counts["G", 2], 0.264705882)
        self.assertAlmostEqual(motif.counts["G", 3], 0.194444444)
        self.assertAlmostEqual(motif.counts["G", 4], 0.102564103)
        self.assertAlmostEqual(motif.counts["G", 5], 0.177777778)
        self.assertAlmostEqual(motif.counts["G", 6], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 7], 0.022222222)
        self.assertAlmostEqual(motif.counts["G", 8], 0.697674419)
        self.assertAlmostEqual(motif.counts["G", 9], 0.571428571)
        self.assertAlmostEqual(motif.counts["G", 10], 0.150000000)
        self.assertAlmostEqual(motif.counts["G", 11], 0.305555556)
        self.assertAlmostEqual(motif.counts["G", 12], 0.258064516)
        self.assertAlmostEqual(motif.counts["G", 13], 0.259259259)
        self.assertAlmostEqual(motif.counts["A", 0], 0.333333333)
        self.assertAlmostEqual(motif.counts["A", 1], 0.103448276)
        self.assertAlmostEqual(motif.counts["A", 2], 0.264705882)
        self.assertAlmostEqual(motif.counts["A", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 4], 0.102564103)
        self.assertAlmostEqual(motif.counts["A", 5], 0.244444444)
        self.assertAlmostEqual(motif.counts["A", 6], 0.800000000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.133333333)
        self.assertAlmostEqual(motif.counts["A", 8], 0.046511628)
        self.assertAlmostEqual(motif.counts["A", 9], 0.238095238)
        self.assertAlmostEqual(motif.counts["A", 10], 0.025000000)
        self.assertAlmostEqual(motif.counts["A", 11], 0.222222222)
        self.assertAlmostEqual(motif.counts["A", 12], 0.354838710)
        self.assertAlmostEqual(motif.counts["A", 13], 0.185185185)
        self.assertAlmostEqual(motif.counts["T", 0], 0.125000000)
        self.assertAlmostEqual(motif.counts["T", 1], 0.103448276)
        self.assertAlmostEqual(motif.counts["T", 2], 0.205882353)
        self.assertAlmostEqual(motif.counts["T", 3], 0.777777778)
        self.assertAlmostEqual(motif.counts["T", 4], 0.743589744)
        self.assertAlmostEqual(motif.counts["T", 5], 0.533333333)
        self.assertAlmostEqual(motif.counts["T", 6], 0.155555556)
        self.assertAlmostEqual(motif.counts["T", 7], 0.688888889)
        self.assertAlmostEqual(motif.counts["T", 8], 0.209302326)
        self.assertAlmostEqual(motif.counts["T", 9], 0.095238095)
        self.assertAlmostEqual(motif.counts["T", 10], 0.025000000)
        self.assertAlmostEqual(motif.counts["T", 11], 0.194444444)
        self.assertAlmostEqual(motif.counts["T", 12], 0.129032258)
        self.assertAlmostEqual(motif.counts["T", 13], 0.222222222)
        self.assertAlmostEqual(motif.counts["C", 0], 0.208333333)
        self.assertAlmostEqual(motif.counts["C", 1], 0.413793103)
        self.assertAlmostEqual(motif.counts["C", 2], 0.264705882)
        self.assertAlmostEqual(motif.counts["C", 3], 0.027777778)
        self.assertAlmostEqual(motif.counts["C", 4], 0.051282051)
        self.assertAlmostEqual(motif.counts["C", 5], 0.044444444)
        self.assertAlmostEqual(motif.counts["C", 6], 0.044444444)
        self.assertAlmostEqual(motif.counts["C", 7], 0.155555556)
        self.assertAlmostEqual(motif.counts["C", 8], 0.046511628)
        self.assertAlmostEqual(motif.counts["C", 9], 0.095238095)
        self.assertAlmostEqual(motif.counts["C", 10], 0.800000000)
        self.assertAlmostEqual(motif.counts["C", 11], 0.277777778)
        self.assertAlmostEqual(motif.counts["C", 12], 0.258064516)
        self.assertAlmostEqual(motif.counts["C", 13], 0.333333333)
        self.assertEqual(motif.consensus, "GCGTTTATGGCGAC")
        self.assertEqual(motif.degenerate_consensus, "NSNTTTATGGCNNN")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.09689283163718865,
                        0.26557323997556864,
                        0.007815379142180268,
                        1.1150033950025815,
                        0.78848108520697,
                        0.3768768552773923,
                        1.125231003810913,
                        0.7023990165752877,
                        0.7536432192801433,
                        0.3995487907017483,
                        1.0658162802208113,
                        0.022422587676774776,
                        0.07979555429087543,
                        0.03400971806422712,
                    ]
                ),
            )
        )
        self.assertEqual(motif[3::2].consensus, "TTTGGC")
        self.assertEqual(motif[3::2].degenerate_consensus, "TTTGNN")
        self.assertTrue(
            np.allclose(
                motif[3::2].relative_entropy,
                np.array(
                    [
                        1.1150033950025815,
                        0.3768768552773923,
                        0.7023990165752877,
                        0.3995487907017483,
                        0.022422587676774776,
                        0.03400971806422712,
                    ]
                ),
            )
        )


class TestJASPAR(unittest.TestCase):

    """Testing parsing JASPAR files."""

    def test_pfm_parsing(self):
        """Test if Bio.motifs can parse JASPAR-style pfm files."""
        with open("motifs/SRF.pfm") as stream:
            m = motifs.read(stream, "pfm")
        self.assertEqual(m.length, 12)

    def test_pfm_four_columns_parsing(self):
        """Test if Bio.motifs.pfm can parse motifs in position frequency matrix format (4 columns)."""
        with open("motifs/fourcolumns.pfm") as stream:
            record = motifs.parse(stream, "pfm-four-columns")
        self.assertEqual(len(record), 8)
        motif = record[0]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 8)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 1], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 2], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 3], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 4], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 5], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 6], 0.009615385)
        self.assertAlmostEqual(motif.counts["G", 7], 0.009615385)
        self.assertAlmostEqual(motif.counts["A", 0], 0.009615385)
        self.assertAlmostEqual(motif.counts["A", 1], 0.009615385)
        self.assertAlmostEqual(motif.counts["A", 2], 0.971153846)
        self.assertAlmostEqual(motif.counts["A", 3], 0.009615385)
        self.assertAlmostEqual(motif.counts["A", 4], 0.009615385)
        self.assertAlmostEqual(motif.counts["A", 5], 0.971153846)
        self.assertAlmostEqual(motif.counts["A", 6], 0.009615385)
        self.assertAlmostEqual(motif.counts["A", 7], 0.009615385)
        self.assertAlmostEqual(motif.counts["T", 0], 0.971153846)
        self.assertAlmostEqual(motif.counts["T", 1], 0.971153846)
        self.assertAlmostEqual(motif.counts["T", 2], 0.009615385)
        self.assertAlmostEqual(motif.counts["T", 3], 0.971153846)
        self.assertAlmostEqual(motif.counts["T", 4], 0.009615385)
        self.assertAlmostEqual(motif.counts["T", 5], 0.009615385)
        self.assertAlmostEqual(motif.counts["T", 6], 0.009615385)
        self.assertAlmostEqual(motif.counts["T", 7], 0.971153846)
        self.assertAlmostEqual(motif.counts["C", 0], 0.009615385)
        self.assertAlmostEqual(motif.counts["C", 1], 0.009615385)
        self.assertAlmostEqual(motif.counts["C", 2], 0.009615385)
        self.assertAlmostEqual(motif.counts["C", 3], 0.009615385)
        self.assertAlmostEqual(motif.counts["C", 4], 0.971153846)
        self.assertAlmostEqual(motif.counts["C", 5], 0.009615385)
        self.assertAlmostEqual(motif.counts["C", 6], 0.971153846)
        self.assertAlmostEqual(motif.counts["C", 7], 0.009615385)
        self.assertEqual(motif.consensus, "TTATCACT")
        self.assertEqual(motif.degenerate_consensus, "TTATCACT")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.765707971839016,
                        1.765707971839016,
                        1.7657079718390165,
                        1.765707971839016,
                        1.7657079718390158,
                        1.7657079718390165,
                        1.7657079718390158,
                        1.765707971839016,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "TATCA")
        motif = record[1]
        self.assertEqual(motif.name, "ENSG00000197372")
        self.assertEqual(motif.length, 20)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.117054000)
        self.assertAlmostEqual(motif.counts["G", 1], 0.364552000)
        self.assertAlmostEqual(motif.counts["G", 2], 0.310520000)
        self.assertAlmostEqual(motif.counts["G", 3], 0.131007000)
        self.assertAlmostEqual(motif.counts["G", 4], 0.176504000)
        self.assertAlmostEqual(motif.counts["G", 5], 0.197793000)
        self.assertAlmostEqual(motif.counts["G", 6], 0.926202000)
        self.assertAlmostEqual(motif.counts["G", 7], 0.983797000)
        self.assertAlmostEqual(motif.counts["G", 8], 0.002387000)
        self.assertAlmostEqual(motif.counts["G", 9], 0.002418000)
        self.assertAlmostEqual(motif.counts["G", 10], 0.001991000)
        self.assertAlmostEqual(motif.counts["G", 11], 0.002868000)
        self.assertAlmostEqual(motif.counts["G", 12], 0.350783000)
        self.assertAlmostEqual(motif.counts["G", 13], 1.000000000)
        self.assertAlmostEqual(motif.counts["G", 14], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 15], 1.000000000)
        self.assertAlmostEqual(motif.counts["G", 16], 1.000000000)
        self.assertAlmostEqual(motif.counts["G", 17], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 18], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 19], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 0], 0.341303000)
        self.assertAlmostEqual(motif.counts["A", 1], 0.283785000)
        self.assertAlmostEqual(motif.counts["A", 2], 0.491055000)
        self.assertAlmostEqual(motif.counts["A", 3], 0.492621000)
        self.assertAlmostEqual(motif.counts["A", 4], 0.250645000)
        self.assertAlmostEqual(motif.counts["A", 5], 0.276694000)
        self.assertAlmostEqual(motif.counts["A", 6], 0.056317000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.004470000)
        self.assertAlmostEqual(motif.counts["A", 8], 0.936213000)
        self.assertAlmostEqual(motif.counts["A", 9], 0.004352000)
        self.assertAlmostEqual(motif.counts["A", 10], 0.013277000)
        self.assertAlmostEqual(motif.counts["A", 11], 0.968132000)
        self.assertAlmostEqual(motif.counts["A", 12], 0.397623000)
        self.assertAlmostEqual(motif.counts["A", 13], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 14], 1.000000000)
        self.assertAlmostEqual(motif.counts["A", 15], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 16], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 17], 1.000000000)
        self.assertAlmostEqual(motif.counts["A", 18], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 19], 1.000000000)
        self.assertAlmostEqual(motif.counts["T", 0], 0.409215000)
        self.assertAlmostEqual(motif.counts["T", 1], 0.274597000)
        self.assertAlmostEqual(motif.counts["T", 2], 0.120217000)
        self.assertAlmostEqual(motif.counts["T", 3], 0.300256000)
        self.assertAlmostEqual(motif.counts["T", 4], 0.211387000)
        self.assertAlmostEqual(motif.counts["T", 5], 0.027444000)
        self.assertAlmostEqual(motif.counts["T", 6], 0.002850000)
        self.assertAlmostEqual(motif.counts["T", 7], 0.003964000)
        self.assertAlmostEqual(motif.counts["T", 8], 0.002613000)
        self.assertAlmostEqual(motif.counts["T", 9], 0.989200000)
        self.assertAlmostEqual(motif.counts["T", 10], 0.976567000)
        self.assertAlmostEqual(motif.counts["T", 11], 0.026737000)
        self.assertAlmostEqual(motif.counts["T", 12], 0.199577000)
        self.assertAlmostEqual(motif.counts["T", 13], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 14], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 15], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 16], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 17], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 18], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 19], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 0], 0.132427000)
        self.assertAlmostEqual(motif.counts["C", 1], 0.077066000)
        self.assertAlmostEqual(motif.counts["C", 2], 0.078208000)
        self.assertAlmostEqual(motif.counts["C", 3], 0.076117000)
        self.assertAlmostEqual(motif.counts["C", 4], 0.361464000)
        self.assertAlmostEqual(motif.counts["C", 5], 0.498070000)
        self.assertAlmostEqual(motif.counts["C", 6], 0.014631000)
        self.assertAlmostEqual(motif.counts["C", 7], 0.007769000)
        self.assertAlmostEqual(motif.counts["C", 8], 0.058787000)
        self.assertAlmostEqual(motif.counts["C", 9], 0.004030000)
        self.assertAlmostEqual(motif.counts["C", 10], 0.008165000)
        self.assertAlmostEqual(motif.counts["C", 11], 0.002263000)
        self.assertAlmostEqual(motif.counts["C", 12], 0.052017000)
        self.assertAlmostEqual(motif.counts["C", 13], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 14], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 15], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 16], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 17], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 18], 1.000000000)
        self.assertAlmostEqual(motif.counts["C", 19], 0.000000000)
        self.assertEqual(motif.consensus, "TGAACCGGATTAAGAGGACA")
        self.assertEqual(motif.degenerate_consensus, "WNRWNMGGATTANGAGGACA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.1946677220077018,
                        0.1566211351816578,
                        0.31728135119311995,
                        0.3086747573287918,
                        0.053393542701508756,
                        0.381471417197324,
                        1.5505596169174871,
                        1.8558501430757017,
                        1.6274200195132635,
                        1.8972899364737197,
                        1.809312450637467,
                        1.7709547585539227,
                        0.2549373240046801,
                        2.0,
                        2.0,
                        2.0,
                        2.0,
                        2.0,
                        2.0,
                        2.0,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "GAACCGGATTAAGAGGA")
        motif = record[2]
        self.assertAlmostEqual(motif.counts["G", 0], 0.083333300)
        self.assertAlmostEqual(motif.counts["G", 1], 0.083333300)
        self.assertAlmostEqual(motif.counts["G", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 4], 0.083333300)
        self.assertAlmostEqual(motif.counts["G", 5], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 6], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 7], 0.333333000)
        self.assertAlmostEqual(motif.counts["G", 8], 0.166667000)
        self.assertAlmostEqual(motif.counts["G", 9], 0.166667000)
        self.assertAlmostEqual(motif.counts["G", 10], 0.416667000)
        self.assertAlmostEqual(motif.counts["A", 0], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 1], 0.750000000)
        self.assertAlmostEqual(motif.counts["A", 2], 0.833333000)
        self.assertAlmostEqual(motif.counts["A", 3], 1.000000000)
        self.assertAlmostEqual(motif.counts["A", 4], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 5], 0.333333000)
        self.assertAlmostEqual(motif.counts["A", 6], 0.833333000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.500000000)
        self.assertAlmostEqual(motif.counts["A", 8], 0.500000000)
        self.assertAlmostEqual(motif.counts["A", 9], 0.333333000)
        self.assertAlmostEqual(motif.counts["A", 10], 0.166667000)
        self.assertAlmostEqual(motif.counts["T", 0], 0.583333000)
        self.assertAlmostEqual(motif.counts["T", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 2], 0.166667000)
        self.assertAlmostEqual(motif.counts["T", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 4], 0.083333300)
        self.assertAlmostEqual(motif.counts["T", 5], 0.666667000)
        self.assertAlmostEqual(motif.counts["T", 6], 0.166667000)
        self.assertAlmostEqual(motif.counts["T", 7], 0.166667000)
        self.assertAlmostEqual(motif.counts["T", 8], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 9], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 10], 0.166667000)
        self.assertAlmostEqual(motif.counts["C", 0], 0.083333300)
        self.assertAlmostEqual(motif.counts["C", 1], 0.166667000)
        self.assertAlmostEqual(motif.counts["C", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 4], 0.833333000)
        self.assertAlmostEqual(motif.counts["C", 5], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 6], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 7], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 8], 0.083333300)
        self.assertAlmostEqual(motif.counts["C", 9], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 10], 0.250000000)
        self.assertEqual(motif.name, "M1734_0.90")
        self.assertEqual(motif.length, 11)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertEqual(motif.consensus, "TAAACTAAAAG")
        self.assertEqual(motif.degenerate_consensus, "TAAACTARNNN")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.4489017067534855,
                        0.9591474871280075,
                        1.3499768043761913,
                        2.0,
                        1.1833109116849791,
                        1.0817044992792044,
                        1.3499768043761913,
                        0.5408517496401433,
                        0.2704258182036411,
                        0.04085174964014324,
                        0.1120812409282564,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "AAACTAAA")
        motif = record[3]
        self.assertEqual(motif.name, "AbdA_Cell_FBgn0000014")
        self.assertEqual(motif.length, 7)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 4], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 5], 6.000000000)
        self.assertAlmostEqual(motif.counts["G", 6], 2.000000000)
        self.assertAlmostEqual(motif.counts["A", 0], 1.000000000)
        self.assertAlmostEqual(motif.counts["A", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 2], 16.000000000)
        self.assertAlmostEqual(motif.counts["A", 3], 18.000000000)
        self.assertAlmostEqual(motif.counts["A", 4], 1.000000000)
        self.assertAlmostEqual(motif.counts["A", 5], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 6], 15.000000000)
        self.assertAlmostEqual(motif.counts["T", 0], 14.000000000)
        self.assertAlmostEqual(motif.counts["T", 1], 18.000000000)
        self.assertAlmostEqual(motif.counts["T", 2], 2.000000000)
        self.assertAlmostEqual(motif.counts["T", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 4], 17.000000000)
        self.assertAlmostEqual(motif.counts["T", 5], 12.000000000)
        self.assertAlmostEqual(motif.counts["T", 6], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 0], 3.000000000)
        self.assertAlmostEqual(motif.counts["C", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 3], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 4], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 5], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 6], 1.000000000)
        self.assertEqual(motif.consensus, "TTAATTA")
        self.assertEqual(motif.degenerate_consensus, "TTAATKA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.0555114658337947,
                        2.0,
                        1.4967416652243541,
                        2.0,
                        1.6904565708496748,
                        1.0817041659455104,
                        1.1969282726758976,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "TAAT")
        motif = record[4]
        self.assertEqual(
            motif.name,
            "ATGACTCATC AP-1(bZIP)/ThioMac-PU.1-ChIP-Seq(GSE21512)/Homer    6.049537    -1.782996e+03   0   9805.3,5781.0,3085.1,2715.0,0.00e+00",
        )
        self.assertEqual(motif.length, 10)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.277000000)
        self.assertAlmostEqual(motif.counts["G", 1], 0.001000000)
        self.assertAlmostEqual(motif.counts["G", 2], 0.965000000)
        self.assertAlmostEqual(motif.counts["G", 3], 0.001000000)
        self.assertAlmostEqual(motif.counts["G", 4], 0.305000000)
        self.assertAlmostEqual(motif.counts["G", 5], 0.001000000)
        self.assertAlmostEqual(motif.counts["G", 6], 0.001000000)
        self.assertAlmostEqual(motif.counts["G", 7], 0.001000000)
        self.assertAlmostEqual(motif.counts["G", 8], 0.307000000)
        self.assertAlmostEqual(motif.counts["G", 9], 0.211000000)
        self.assertAlmostEqual(motif.counts["A", 0], 0.419000000)
        self.assertAlmostEqual(motif.counts["A", 1], 0.001000000)
        self.assertAlmostEqual(motif.counts["A", 2], 0.010000000)
        self.assertAlmostEqual(motif.counts["A", 3], 0.984000000)
        self.assertAlmostEqual(motif.counts["A", 4], 0.062000000)
        self.assertAlmostEqual(motif.counts["A", 5], 0.026000000)
        self.assertAlmostEqual(motif.counts["A", 6], 0.043000000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.980000000)
        self.assertAlmostEqual(motif.counts["A", 8], 0.050000000)
        self.assertAlmostEqual(motif.counts["A", 9], 0.149000000)
        self.assertAlmostEqual(motif.counts["T", 0], 0.028000000)
        self.assertAlmostEqual(motif.counts["T", 1], 0.997000000)
        self.assertAlmostEqual(motif.counts["T", 2], 0.023000000)
        self.assertAlmostEqual(motif.counts["T", 3], 0.012000000)
        self.assertAlmostEqual(motif.counts["T", 4], 0.054000000)
        self.assertAlmostEqual(motif.counts["T", 5], 0.972000000)
        self.assertAlmostEqual(motif.counts["T", 6], 0.012000000)
        self.assertAlmostEqual(motif.counts["T", 7], 0.014000000)
        self.assertAlmostEqual(motif.counts["T", 8], 0.471000000)
        self.assertAlmostEqual(motif.counts["T", 9], 0.195000000)
        self.assertAlmostEqual(motif.counts["C", 0], 0.275000000)
        self.assertAlmostEqual(motif.counts["C", 1], 0.001000000)
        self.assertAlmostEqual(motif.counts["C", 2], 0.002000000)
        self.assertAlmostEqual(motif.counts["C", 3], 0.003000000)
        self.assertAlmostEqual(motif.counts["C", 4], 0.579000000)
        self.assertAlmostEqual(motif.counts["C", 5], 0.001000000)
        self.assertAlmostEqual(motif.counts["C", 6], 0.943000000)
        self.assertAlmostEqual(motif.counts["C", 7], 0.005000000)
        self.assertAlmostEqual(motif.counts["C", 8], 0.172000000)
        self.assertAlmostEqual(motif.counts["C", 9], 0.444000000)
        self.assertEqual(motif.consensus, "ATGACTCATC")
        self.assertEqual(motif.degenerate_consensus, "NTGASTCAKN")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.30427230622817475,
                        1.9657810606529142,
                        1.7408585738061,
                        1.8654244261025423,
                        0.5449286810918202,
                        1.8033449015144003,
                        1.639502374827662,
                        1.8370335049436752,
                        0.3124728907316759,
                        0.13671828556764112,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "TGACTCA")
        motif = record[5]
        self.assertEqual(motif.name, "AHR_si")
        self.assertEqual(motif.length, 9)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 56.412537571)
        self.assertAlmostEqual(motif.counts["G", 1], 34.663129823)
        self.assertAlmostEqual(motif.counts["G", 2], 20.706746562)
        self.assertAlmostEqual(motif.counts["G", 3], 145.863705132)
        self.assertAlmostEqual(motif.counts["G", 4], 1.492783630)
        self.assertAlmostEqual(motif.counts["G", 5], 149.376137203)
        self.assertAlmostEqual(motif.counts["G", 6], 0.702486414)
        self.assertAlmostEqual(motif.counts["G", 7], 153.958717377)
        self.assertAlmostEqual(motif.counts["G", 8], 16.159862547)
        self.assertAlmostEqual(motif.counts["A", 0], 40.513432405)
        self.assertAlmostEqual(motif.counts["A", 1], 10.877470983)
        self.assertAlmostEqual(motif.counts["A", 2], 21.716570782)
        self.assertAlmostEqual(motif.counts["A", 3], 2.546513251)
        self.assertAlmostEqual(motif.counts["A", 4], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 5], 3.441039751)
        self.assertAlmostEqual(motif.counts["A", 6], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 8], 43.079223333)
        self.assertAlmostEqual(motif.counts["T", 0], 38.773634853)
        self.assertAlmostEqual(motif.counts["T", 1], 96.547239851)
        self.assertAlmostEqual(motif.counts["T", 2], 67.652320196)
        self.assertAlmostEqual(motif.counts["T", 3], 4.231336967)
        self.assertAlmostEqual(motif.counts["T", 4], 2.107459242)
        self.assertAlmostEqual(motif.counts["T", 5], 0.351243207)
        self.assertAlmostEqual(motif.counts["T", 6], 149.815191211)
        self.assertAlmostEqual(motif.counts["T", 7], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 8], 27.844049228)
        self.assertAlmostEqual(motif.counts["C", 0], 18.259112548)
        self.assertAlmostEqual(motif.counts["C", 1], 11.870876720)
        self.assertAlmostEqual(motif.counts["C", 2], 43.883079838)
        self.assertAlmostEqual(motif.counts["C", 3], 1.317162026)
        self.assertAlmostEqual(motif.counts["C", 4], 150.358474505)
        self.assertAlmostEqual(motif.counts["C", 5], 0.790297216)
        self.assertAlmostEqual(motif.counts["C", 6], 3.441039751)
        self.assertAlmostEqual(motif.counts["C", 7], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 8], 66.875582269)
        self.assertEqual(motif.consensus, "GTTGCGTGC")
        self.assertEqual(motif.degenerate_consensus, "NTNGCGTGN")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.09662409645348236,
                        0.5383413903068038,
                        0.17471270188228985,
                        1.6270151623731723,
                        1.8170663607301638,
                        1.7760800937680195,
                        1.803660112630464,
                        2.0,
                        0.17577786614573548,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "TTGCGT")
        motif = record[6]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 8)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.098612000)
        self.assertAlmostEqual(motif.counts["G", 1], 0.025056000)
        self.assertAlmostEqual(motif.counts["G", 2], 0.918728000)
        self.assertAlmostEqual(motif.counts["G", 3], 0.029759000)
        self.assertAlmostEqual(motif.counts["G", 4], 0.104968000)
        self.assertAlmostEqual(motif.counts["G", 5], 0.006667000)
        self.assertAlmostEqual(motif.counts["G", 6], 0.026928000)
        self.assertAlmostEqual(motif.counts["G", 7], 0.005737000)
        self.assertAlmostEqual(motif.counts["A", 0], 0.772949000)
        self.assertAlmostEqual(motif.counts["A", 1], 0.026652000)
        self.assertAlmostEqual(motif.counts["A", 2], 0.017663000)
        self.assertAlmostEqual(motif.counts["A", 3], 0.919596000)
        self.assertAlmostEqual(motif.counts["A", 4], 0.060312000)
        self.assertAlmostEqual(motif.counts["A", 5], 0.037406000)
        self.assertAlmostEqual(motif.counts["A", 6], 0.047316000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.948639000)
        self.assertAlmostEqual(motif.counts["T", 0], 0.038860000)
        self.assertAlmostEqual(motif.counts["T", 1], 0.943639000)
        self.assertAlmostEqual(motif.counts["T", 2], 0.040264000)
        self.assertAlmostEqual(motif.counts["T", 3], 0.025231000)
        self.assertAlmostEqual(motif.counts["T", 4], 0.062462000)
        self.assertAlmostEqual(motif.counts["T", 5], 0.935284000)
        self.assertAlmostEqual(motif.counts["T", 6], 0.026732000)
        self.assertAlmostEqual(motif.counts["T", 7], 0.026128000)
        self.assertAlmostEqual(motif.counts["C", 0], 0.089579000)
        self.assertAlmostEqual(motif.counts["C", 1], 0.004653000)
        self.assertAlmostEqual(motif.counts["C", 2], 0.023344000)
        self.assertAlmostEqual(motif.counts["C", 3], 0.025414000)
        self.assertAlmostEqual(motif.counts["C", 4], 0.772259000)
        self.assertAlmostEqual(motif.counts["C", 5], 0.020643000)
        self.assertAlmostEqual(motif.counts["C", 6], 0.899024000)
        self.assertAlmostEqual(motif.counts["C", 7], 0.019497000)
        self.assertEqual(motif.consensus, "ATGACTCA")
        self.assertEqual(motif.degenerate_consensus, "ATGACTCA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.889358068874075,
                        1.6123293058245811,
                        1.471654165929799,
                        1.4693092198124151,
                        0.8764628815119266,
                        1.5686388858173408,
                        1.37357038822754,
                        1.6369796776980579,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "TGACT")
        motif = record[7]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 11)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 28.0)
        self.assertAlmostEqual(motif.counts["G", 1], 0.0)
        self.assertAlmostEqual(motif.counts["G", 2], 14.0)
        self.assertAlmostEqual(motif.counts["G", 3], 0.0)
        self.assertAlmostEqual(motif.counts["G", 4], 0.0)
        self.assertAlmostEqual(motif.counts["G", 5], 7.0)
        self.assertAlmostEqual(motif.counts["G", 6], 11.0)
        self.assertAlmostEqual(motif.counts["G", 7], 38.0)
        self.assertAlmostEqual(motif.counts["G", 8], 0.0)
        self.assertAlmostEqual(motif.counts["G", 9], 25.0)
        self.assertAlmostEqual(motif.counts["G", 10], 0.0)
        self.assertAlmostEqual(motif.counts["A", 0], 0.0)
        self.assertAlmostEqual(motif.counts["A", 1], 0.0)
        self.assertAlmostEqual(motif.counts["A", 2], 55.0)
        self.assertAlmostEqual(motif.counts["A", 3], 99.0)
        self.assertAlmostEqual(motif.counts["A", 4], 78.0)
        self.assertAlmostEqual(motif.counts["A", 5], 52.0)
        self.assertAlmostEqual(motif.counts["A", 6], 46.0)
        self.assertAlmostEqual(motif.counts["A", 7], 60.0)
        self.assertAlmostEqual(motif.counts["A", 8], 33.0)
        self.assertAlmostEqual(motif.counts["A", 9], 0.0)
        self.assertAlmostEqual(motif.counts["A", 10], 0.0)
        self.assertAlmostEqual(motif.counts["T", 0], 30.0)
        self.assertAlmostEqual(motif.counts["T", 1], 0.0)
        self.assertAlmostEqual(motif.counts["T", 2], 0.0)
        self.assertAlmostEqual(motif.counts["T", 3], 0.0)
        self.assertAlmostEqual(motif.counts["T", 4], 20.0)
        self.assertAlmostEqual(motif.counts["T", 5], 0.0)
        self.assertAlmostEqual(motif.counts["T", 6], 19.0)
        self.assertAlmostEqual(motif.counts["T", 7], 0.0)
        self.assertAlmostEqual(motif.counts["T", 8], 0.0)
        self.assertAlmostEqual(motif.counts["T", 9], 73.0)
        self.assertAlmostEqual(motif.counts["T", 10], 99.0)
        self.assertAlmostEqual(motif.counts["C", 0], 40.0)
        self.assertAlmostEqual(motif.counts["C", 1], 99.0)
        self.assertAlmostEqual(motif.counts["C", 2], 29.0)
        self.assertAlmostEqual(motif.counts["C", 3], 0.0)
        self.assertAlmostEqual(motif.counts["C", 4], 0.0)
        self.assertAlmostEqual(motif.counts["C", 5], 39.0)
        self.assertAlmostEqual(motif.counts["C", 6], 22.0)
        self.assertAlmostEqual(motif.counts["C", 7], 0.0)
        self.assertAlmostEqual(motif.counts["C", 8], 66.0)
        self.assertAlmostEqual(motif.counts["C", 9], 0.0)
        self.assertAlmostEqual(motif.counts["C", 10], 0.0)
        self.assertEqual(motif.consensus, "CCAAAAAACTT")
        self.assertEqual(motif.degenerate_consensus, "BCMAAMNRMTT")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.43314504855176084,
                        2.0,
                        0.6114044621231828,
                        2.0,
                        1.2699833698542062,
                        0.7139129756130338,
                        0.1909607288346033,
                        1.0366644543273158,
                        1.0817041659455104,
                        1.180735028768561,
                        2.0,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].consensus, "CAAAAAAC")

    def test_pfm_four_rows_parsing(self):
        """Test if Bio.motifs.pfm can parse motifs in position frequency matrix format (4 rows)."""
        with open("motifs/fourrows.pfm") as stream:
            record = motifs.parse(stream, "pfm-four-rows")
        self.assertEqual(len(record), 9)
        motif = record[0]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 6)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 5.0)
        self.assertAlmostEqual(motif.counts["G", 1], 0.0)
        self.assertAlmostEqual(motif.counts["G", 2], 0.0)
        self.assertAlmostEqual(motif.counts["G", 3], 0.0)
        self.assertAlmostEqual(motif.counts["G", 4], 3.0)
        self.assertAlmostEqual(motif.counts["G", 5], 0.0)
        self.assertAlmostEqual(motif.counts["A", 0], 0.0)
        self.assertAlmostEqual(motif.counts["A", 1], 5.0)
        self.assertAlmostEqual(motif.counts["A", 2], 6.0)
        self.assertAlmostEqual(motif.counts["A", 3], 5.0)
        self.assertAlmostEqual(motif.counts["A", 4], 1.0)
        self.assertAlmostEqual(motif.counts["A", 5], 0.0)
        self.assertAlmostEqual(motif.counts["T", 0], 0.0)
        self.assertAlmostEqual(motif.counts["T", 1], 0.0)
        self.assertAlmostEqual(motif.counts["T", 2], 0.0)
        self.assertAlmostEqual(motif.counts["T", 3], 1.0)
        self.assertAlmostEqual(motif.counts["T", 4], 2.0)
        self.assertAlmostEqual(motif.counts["T", 5], 2.0)
        self.assertAlmostEqual(motif.counts["C", 0], 1.0)
        self.assertAlmostEqual(motif.counts["C", 1], 1.0)
        self.assertAlmostEqual(motif.counts["C", 2], 0.0)
        self.assertAlmostEqual(motif.counts["C", 3], 0.0)
        self.assertAlmostEqual(motif.counts["C", 4], 0.0)
        self.assertAlmostEqual(motif.counts["C", 5], 4.0)
        self.assertEqual(motif.consensus, "GAAAGC")
        self.assertEqual(motif.degenerate_consensus, "GAAAKY")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.349977578351646,
                        1.349977578351646,
                        2.0,
                        1.349977578351646,
                        0.5408520829727552,
                        1.0817041659455104,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "GAAA")
        motif = record[1]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 15)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 1], 1.000000000)
        self.assertAlmostEqual(motif.counts["G", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 3], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 4], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 5], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 6], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 7], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 8], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 9], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 10], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 11], 0.250000000)
        self.assertAlmostEqual(motif.counts["G", 12], 0.000000000)
        self.assertAlmostEqual(motif.counts["G", 13], 1.000000000)
        self.assertAlmostEqual(motif.counts["G", 14], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 0], 0.500000000)
        self.assertAlmostEqual(motif.counts["A", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 3], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 4], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 5], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 6], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 7], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 8], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 9], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 10], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 11], 0.250000000)
        self.assertAlmostEqual(motif.counts["A", 12], 0.500000000)
        self.assertAlmostEqual(motif.counts["A", 13], 0.000000000)
        self.assertAlmostEqual(motif.counts["A", 14], 0.083333333)
        self.assertAlmostEqual(motif.counts["T", 0], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 2], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 3], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 4], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 5], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 6], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 7], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 8], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 9], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 10], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 11], 0.250000000)
        self.assertAlmostEqual(motif.counts["T", 12], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 13], 0.000000000)
        self.assertAlmostEqual(motif.counts["T", 14], 0.083333333)
        self.assertAlmostEqual(motif.counts["C", 0], 0.500000000)
        self.assertAlmostEqual(motif.counts["C", 1], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 2], 1.000000000)
        self.assertAlmostEqual(motif.counts["C", 3], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 4], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 5], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 6], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 7], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 8], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 9], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 10], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 11], 0.250000000)
        self.assertAlmostEqual(motif.counts["C", 12], 0.500000000)
        self.assertAlmostEqual(motif.counts["C", 13], 0.000000000)
        self.assertAlmostEqual(motif.counts["C", 14], 0.583333333)
        self.assertEqual(motif.consensus, "AGCGGGGGGGGGAGC")
        self.assertEqual(motif.degenerate_consensus, "MGCNNNNNNNNNMGC")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.0,
                        2.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        2.0,
                        0.44890182844369547,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "AGCGGGGGGGGGA")
        motif = record[2]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 15)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 270.0)
        self.assertAlmostEqual(motif.counts["G", 1], 398.0)
        self.assertAlmostEqual(motif.counts["G", 2], 54.0)
        self.assertAlmostEqual(motif.counts["G", 3], 164.0)
        self.assertAlmostEqual(motif.counts["G", 4], 7.0)
        self.assertAlmostEqual(motif.counts["G", 5], 659.0)
        self.assertAlmostEqual(motif.counts["G", 6], 1.0)
        self.assertAlmostEqual(motif.counts["G", 7], 750.0)
        self.assertAlmostEqual(motif.counts["G", 8], 755.0)
        self.assertAlmostEqual(motif.counts["G", 9], 65.0)
        self.assertAlmostEqual(motif.counts["G", 10], 1.0)
        self.assertAlmostEqual(motif.counts["G", 11], 41.0)
        self.assertAlmostEqual(motif.counts["G", 12], 202.0)
        self.assertAlmostEqual(motif.counts["G", 13], 234.0)
        self.assertAlmostEqual(motif.counts["G", 14], 205.0)
        self.assertAlmostEqual(motif.counts["A", 0], 92.0)
        self.assertAlmostEqual(motif.counts["A", 1], 106.0)
        self.assertAlmostEqual(motif.counts["A", 2], 231.0)
        self.assertAlmostEqual(motif.counts["A", 3], 135.0)
        self.assertAlmostEqual(motif.counts["A", 4], 0.0)
        self.assertAlmostEqual(motif.counts["A", 5], 1.0)
        self.assertAlmostEqual(motif.counts["A", 6], 780.0)
        self.assertAlmostEqual(motif.counts["A", 7], 28.0)
        self.assertAlmostEqual(motif.counts["A", 8], 0.0)
        self.assertAlmostEqual(motif.counts["A", 9], 700.0)
        self.assertAlmostEqual(motif.counts["A", 10], 739.0)
        self.assertAlmostEqual(motif.counts["A", 11], 94.0)
        self.assertAlmostEqual(motif.counts["A", 12], 60.0)
        self.assertAlmostEqual(motif.counts["A", 13], 127.0)
        self.assertAlmostEqual(motif.counts["A", 14], 130.0)
        self.assertAlmostEqual(motif.counts["T", 0], 290.0)
        self.assertAlmostEqual(motif.counts["T", 1], 204.0)
        self.assertAlmostEqual(motif.counts["T", 2], 375.0)
        self.assertAlmostEqual(motif.counts["T", 3], 411.0)
        self.assertAlmostEqual(motif.counts["T", 4], 9.0)
        self.assertAlmostEqual(motif.counts["T", 5], 127.0)
        self.assertAlmostEqual(motif.counts["T", 6], 6.0)
        self.assertAlmostEqual(motif.counts["T", 7], 11.0)
        self.assertAlmostEqual(motif.counts["T", 8], 36.0)
        self.assertAlmostEqual(motif.counts["T", 9], 20.0)
        self.assertAlmostEqual(motif.counts["T", 10], 31.0)
        self.assertAlmostEqual(motif.counts["T", 11], 605.0)
        self.assertAlmostEqual(motif.counts["T", 12], 335.0)
        self.assertAlmostEqual(motif.counts["T", 13], 307.0)
        self.assertAlmostEqual(motif.counts["T", 14], 308.0)
        self.assertAlmostEqual(motif.counts["C", 0], 138.0)
        self.assertAlmostEqual(motif.counts["C", 1], 82.0)
        self.assertAlmostEqual(motif.counts["C", 2], 129.0)
        self.assertAlmostEqual(motif.counts["C", 3], 81.0)
        self.assertAlmostEqual(motif.counts["C", 4], 774.0)
        self.assertAlmostEqual(motif.counts["C", 5], 1.0)
        self.assertAlmostEqual(motif.counts["C", 6], 3.0)
        self.assertAlmostEqual(motif.counts["C", 7], 1.0)
        self.assertAlmostEqual(motif.counts["C", 8], 0.0)
        self.assertAlmostEqual(motif.counts["C", 9], 6.0)
        self.assertAlmostEqual(motif.counts["C", 10], 17.0)
        self.assertAlmostEqual(motif.counts["C", 11], 49.0)
        self.assertAlmostEqual(motif.counts["C", 12], 193.0)
        self.assertAlmostEqual(motif.counts["C", 13], 122.0)
        self.assertAlmostEqual(motif.counts["C", 14], 148.0)
        self.assertEqual(motif.consensus, "TGTTCGAGGAATTTT")
        self.assertEqual(motif.degenerate_consensus, "NKWTCGAGGAATNNN")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.13892143832881046,
                        0.2692660952911542,
                        0.27915566353819243,
                        0.2665840150038887,
                        1.8371160692433293,
                        1.3354706334248059,
                        1.8856611660889357,
                        1.6600123906824402,
                        1.7329826640509962,
                        1.3601399752384014,
                        1.5978925123167893,
                        0.8698961051280728,
                        0.19290147849975406,
                        0.11003972948477392,
                        0.08469189143040626,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "TGTTCGAGGAATT")
        motif = record[3]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 6)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 2.0)
        self.assertAlmostEqual(motif.counts["G", 1], 1.0)
        self.assertAlmostEqual(motif.counts["G", 2], 1.0)
        self.assertAlmostEqual(motif.counts["G", 3], 1.0)
        self.assertAlmostEqual(motif.counts["G", 4], 97.0)
        self.assertAlmostEqual(motif.counts["G", 5], 2.0)
        self.assertAlmostEqual(motif.counts["A", 0], 9.0)
        self.assertAlmostEqual(motif.counts["A", 1], 1.0)
        self.assertAlmostEqual(motif.counts["A", 2], 1.0)
        self.assertAlmostEqual(motif.counts["A", 3], 97.0)
        self.assertAlmostEqual(motif.counts["A", 4], 1.0)
        self.assertAlmostEqual(motif.counts["A", 5], 94.0)
        self.assertAlmostEqual(motif.counts["T", 0], 80.0)
        self.assertAlmostEqual(motif.counts["T", 1], 1.0)
        self.assertAlmostEqual(motif.counts["T", 2], 97.0)
        self.assertAlmostEqual(motif.counts["T", 3], 1.0)
        self.assertAlmostEqual(motif.counts["T", 4], 1.0)
        self.assertAlmostEqual(motif.counts["T", 5], 2.0)
        self.assertAlmostEqual(motif.counts["C", 0], 9.0)
        self.assertAlmostEqual(motif.counts["C", 1], 97.0)
        self.assertAlmostEqual(motif.counts["C", 2], 1.0)
        self.assertAlmostEqual(motif.counts["C", 3], 1.0)
        self.assertAlmostEqual(motif.counts["C", 4], 1.0)
        self.assertAlmostEqual(motif.counts["C", 5], 2.0)
        self.assertEqual(motif.consensus, "TCTAGA")
        self.assertEqual(motif.degenerate_consensus, "TCTAGA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.0042727863947818,
                        1.758059267146789,
                        1.7580592671467892,
                        1.7580592671467892,
                        1.7580592671467892,
                        1.5774573308022544,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "TCTA")
        motif = record[4]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 6)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.02)
        self.assertAlmostEqual(motif.counts["G", 1], 0.01)
        self.assertAlmostEqual(motif.counts["G", 2], 0.01)
        self.assertAlmostEqual(motif.counts["G", 3], 0.01)
        self.assertAlmostEqual(motif.counts["G", 4], 0.97)
        self.assertAlmostEqual(motif.counts["G", 5], 0.02)
        self.assertAlmostEqual(motif.counts["A", 0], 0.09)
        self.assertAlmostEqual(motif.counts["A", 1], 0.01)
        self.assertAlmostEqual(motif.counts["A", 2], 0.01)
        self.assertAlmostEqual(motif.counts["A", 3], 0.97)
        self.assertAlmostEqual(motif.counts["A", 4], 0.01)
        self.assertAlmostEqual(motif.counts["A", 5], 0.94)
        self.assertAlmostEqual(motif.counts["T", 0], 0.80)
        self.assertAlmostEqual(motif.counts["T", 1], 0.01)
        self.assertAlmostEqual(motif.counts["T", 2], 0.97)
        self.assertAlmostEqual(motif.counts["T", 3], 0.01)
        self.assertAlmostEqual(motif.counts["T", 4], 0.01)
        self.assertAlmostEqual(motif.counts["T", 5], 0.02)
        self.assertAlmostEqual(motif.counts["C", 0], 0.09)
        self.assertAlmostEqual(motif.counts["C", 1], 0.97)
        self.assertAlmostEqual(motif.counts["C", 2], 0.01)
        self.assertAlmostEqual(motif.counts["C", 3], 0.01)
        self.assertAlmostEqual(motif.counts["C", 4], 0.01)
        self.assertAlmostEqual(motif.counts["C", 5], 0.02)
        self.assertEqual(motif.consensus, "TCTAGA")
        self.assertEqual(motif.degenerate_consensus, "TCTAGA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.0042727863947818,
                        1.758059267146789,
                        1.7580592671467892,
                        1.7580592671467892,
                        1.7580592671467892,
                        1.5774573308022544,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "TCTA")
        motif = record[5]
        self.assertEqual(motif.name, "abd-A")
        self.assertEqual(motif.length, 8)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.455991516)
        self.assertAlmostEqual(motif.counts["G", 1], 0.069194062)
        self.assertAlmostEqual(motif.counts["G", 2], 0.010869565)
        self.assertAlmostEqual(motif.counts["G", 3], 0.021739130)
        self.assertAlmostEqual(motif.counts["G", 4], 0.028499470)
        self.assertAlmostEqual(motif.counts["G", 5], 0.028499470)
        self.assertAlmostEqual(motif.counts["G", 6], 0.016304348)
        self.assertAlmostEqual(motif.counts["G", 7], 0.160127253)
        self.assertAlmostEqual(motif.counts["A", 0], 0.218451750)
        self.assertAlmostEqual(motif.counts["A", 1], 0.023064687)
        self.assertAlmostEqual(motif.counts["A", 2], 0.656680806)
        self.assertAlmostEqual(motif.counts["A", 3], 0.898197243)
        self.assertAlmostEqual(motif.counts["A", 4], 0.040694592)
        self.assertAlmostEqual(motif.counts["A", 5], 0.132953340)
        self.assertAlmostEqual(motif.counts["A", 6], 0.749072110)
        self.assertAlmostEqual(motif.counts["A", 7], 0.628313892)
        self.assertAlmostEqual(motif.counts["T", 0], 0.235949099)
        self.assertAlmostEqual(motif.counts["T", 1], 0.590402969)
        self.assertAlmostEqual(motif.counts["T", 2], 0.010869565)
        self.assertAlmostEqual(motif.counts["T", 3], 0.033934252)
        self.assertAlmostEqual(motif.counts["T", 4], 0.880567338)
        self.assertAlmostEqual(motif.counts["T", 5], 0.797852598)
        self.assertAlmostEqual(motif.counts["T", 6], 0.206124072)
        self.assertAlmostEqual(motif.counts["T", 7], 0.177624602)
        self.assertAlmostEqual(motif.counts["C", 0], 0.089607635)
        self.assertAlmostEqual(motif.counts["C", 1], 0.317338282)
        self.assertAlmostEqual(motif.counts["C", 2], 0.321580064)
        self.assertAlmostEqual(motif.counts["C", 3], 0.046129374)
        self.assertAlmostEqual(motif.counts["C", 4], 0.050238600)
        self.assertAlmostEqual(motif.counts["C", 5], 0.040694592)
        self.assertAlmostEqual(motif.counts["C", 6], 0.028499470)
        self.assertAlmostEqual(motif.counts["C", 7], 0.033934252)
        self.assertEqual(motif.consensus, "GTAATTAA")
        self.assertEqual(motif.degenerate_consensus, "NYAATTAA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.2005361303021225,
                        0.6336277209668335,
                        0.933405467206956,
                        1.3704286046679186,
                        1.2873833086962072,
                        1.0187720746919493,
                        0.975022432438911,
                        0.547109562258496,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "GTAATT")
        motif = record[6]
        self.assertEqual(motif.name, "MA0001.1 AGL3")
        self.assertEqual(motif.length, 10)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 1.0)
        self.assertAlmostEqual(motif.counts["G", 1], 0.0)
        self.assertAlmostEqual(motif.counts["G", 2], 3.0)
        self.assertAlmostEqual(motif.counts["G", 3], 4.0)
        self.assertAlmostEqual(motif.counts["G", 4], 1.0)
        self.assertAlmostEqual(motif.counts["G", 5], 0.0)
        self.assertAlmostEqual(motif.counts["G", 6], 5.0)
        self.assertAlmostEqual(motif.counts["G", 7], 3.0)
        self.assertAlmostEqual(motif.counts["G", 8], 28.0)
        self.assertAlmostEqual(motif.counts["G", 9], 88.0)
        self.assertAlmostEqual(motif.counts["A", 0], 0.0)
        self.assertAlmostEqual(motif.counts["A", 1], 3.0)
        self.assertAlmostEqual(motif.counts["A", 2], 79.0)
        self.assertAlmostEqual(motif.counts["A", 3], 40.0)
        self.assertAlmostEqual(motif.counts["A", 4], 66.0)
        self.assertAlmostEqual(motif.counts["A", 5], 48.0)
        self.assertAlmostEqual(motif.counts["A", 6], 65.0)
        self.assertAlmostEqual(motif.counts["A", 7], 11.0)
        self.assertAlmostEqual(motif.counts["A", 8], 65.0)
        self.assertAlmostEqual(motif.counts["A", 9], 0.0)
        self.assertAlmostEqual(motif.counts["T", 0], 2.0)
        self.assertAlmostEqual(motif.counts["T", 1], 19.0)
        self.assertAlmostEqual(motif.counts["T", 2], 11.0)
        self.assertAlmostEqual(motif.counts["T", 3], 50.0)
        self.assertAlmostEqual(motif.counts["T", 4], 29.0)
        self.assertAlmostEqual(motif.counts["T", 5], 47.0)
        self.assertAlmostEqual(motif.counts["T", 6], 22.0)
        self.assertAlmostEqual(motif.counts["T", 7], 81.0)
        self.assertAlmostEqual(motif.counts["T", 8], 1.0)
        self.assertAlmostEqual(motif.counts["T", 9], 6.0)
        self.assertAlmostEqual(motif.counts["C", 0], 94.0)
        self.assertAlmostEqual(motif.counts["C", 1], 75.0)
        self.assertAlmostEqual(motif.counts["C", 2], 4.0)
        self.assertAlmostEqual(motif.counts["C", 3], 3.0)
        self.assertAlmostEqual(motif.counts["C", 4], 1.0)
        self.assertAlmostEqual(motif.counts["C", 5], 2.0)
        self.assertAlmostEqual(motif.counts["C", 6], 5.0)
        self.assertAlmostEqual(motif.counts["C", 7], 2.0)
        self.assertAlmostEqual(motif.counts["C", 8], 3.0)
        self.assertAlmostEqual(motif.counts["C", 9], 3.0)
        self.assertEqual(motif.consensus, "CCATAAATAG")
        self.assertEqual(motif.degenerate_consensus, "CCAWAWATAG")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.7725753233561499,
                        1.0972718180683638,
                        1.0578945228970464,
                        0.6353945886004412,
                        0.9651537633423314,
                        0.8757972203228152,
                        0.6864859661195083,
                        1.1561334005018244,
                        0.8724039945822116,
                        1.4691041160249607,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "CCATAAAT")
        motif = record[7]
        self.assertEqual(motif.name, "MA0001.1 AGL3")
        self.assertEqual(motif.length, 10)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 1.0)
        self.assertAlmostEqual(motif.counts["G", 1], 0.0)
        self.assertAlmostEqual(motif.counts["G", 2], 3.0)
        self.assertAlmostEqual(motif.counts["G", 3], 4.0)
        self.assertAlmostEqual(motif.counts["G", 4], 1.0)
        self.assertAlmostEqual(motif.counts["G", 5], 0.0)
        self.assertAlmostEqual(motif.counts["G", 6], 5.0)
        self.assertAlmostEqual(motif.counts["G", 7], 3.0)
        self.assertAlmostEqual(motif.counts["G", 8], 28.0)
        self.assertAlmostEqual(motif.counts["G", 9], 88.0)
        self.assertAlmostEqual(motif.counts["A", 0], 0.0)
        self.assertAlmostEqual(motif.counts["A", 1], 3.0)
        self.assertAlmostEqual(motif.counts["A", 2], 79.0)
        self.assertAlmostEqual(motif.counts["A", 3], 40.0)
        self.assertAlmostEqual(motif.counts["A", 4], 66.0)
        self.assertAlmostEqual(motif.counts["A", 5], 48.0)
        self.assertAlmostEqual(motif.counts["A", 6], 65.0)
        self.assertAlmostEqual(motif.counts["A", 7], 11.0)
        self.assertAlmostEqual(motif.counts["A", 8], 65.0)
        self.assertAlmostEqual(motif.counts["A", 9], 0.0)
        self.assertAlmostEqual(motif.counts["T", 0], 2.0)
        self.assertAlmostEqual(motif.counts["T", 1], 19.0)
        self.assertAlmostEqual(motif.counts["T", 2], 11.0)
        self.assertAlmostEqual(motif.counts["T", 3], 50.0)
        self.assertAlmostEqual(motif.counts["T", 4], 29.0)
        self.assertAlmostEqual(motif.counts["T", 5], 47.0)
        self.assertAlmostEqual(motif.counts["T", 6], 22.0)
        self.assertAlmostEqual(motif.counts["T", 7], 81.0)
        self.assertAlmostEqual(motif.counts["T", 8], 1.0)
        self.assertAlmostEqual(motif.counts["T", 9], 6.0)
        self.assertAlmostEqual(motif.counts["C", 0], 94.0)
        self.assertAlmostEqual(motif.counts["C", 1], 75.0)
        self.assertAlmostEqual(motif.counts["C", 2], 4.0)
        self.assertAlmostEqual(motif.counts["C", 3], 3.0)
        self.assertAlmostEqual(motif.counts["C", 4], 1.0)
        self.assertAlmostEqual(motif.counts["C", 5], 2.0)
        self.assertAlmostEqual(motif.counts["C", 6], 5.0)
        self.assertAlmostEqual(motif.counts["C", 7], 2.0)
        self.assertAlmostEqual(motif.counts["C", 8], 3.0)
        self.assertAlmostEqual(motif.counts["C", 9], 3.0)
        self.assertEqual(motif.consensus, "CCATAAATAG")
        self.assertEqual(motif.degenerate_consensus, "CCAWAWATAG")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.7725753233561499,
                        1.0972718180683638,
                        1.0578945228970464,
                        0.6353945886004412,
                        0.9651537633423314,
                        0.8757972203228152,
                        0.6864859661195083,
                        1.1561334005018244,
                        0.8724039945822116,
                        1.4691041160249607,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "CCATAAAT")
        motif = record[8]
        self.assertEqual(motif.name, "")
        self.assertEqual(motif.length, 9)
        self.assertEqual(motif.alphabet, "GATC")
        self.assertAlmostEqual(motif.counts["G", 0], 0.016)
        self.assertAlmostEqual(motif.counts["G", 1], 0.020)
        self.assertAlmostEqual(motif.counts["G", 2], 0.028)
        self.assertAlmostEqual(motif.counts["G", 3], 0.016)
        self.assertAlmostEqual(motif.counts["G", 4], 0.020)
        self.assertAlmostEqual(motif.counts["G", 5], 0.028)
        self.assertAlmostEqual(motif.counts["G", 6], 0.047)
        self.assertAlmostEqual(motif.counts["G", 7], 0.045)
        self.assertAlmostEqual(motif.counts["G", 8], 0.216)
        self.assertAlmostEqual(motif.counts["A", 0], 0.116)
        self.assertAlmostEqual(motif.counts["A", 1], 0.974)
        self.assertAlmostEqual(motif.counts["A", 2], 0.444)
        self.assertAlmostEqual(motif.counts["A", 3], 0.116)
        self.assertAlmostEqual(motif.counts["A", 4], 0.974)
        self.assertAlmostEqual(motif.counts["A", 5], 0.444)
        self.assertAlmostEqual(motif.counts["A", 6], 0.667)
        self.assertAlmostEqual(motif.counts["A", 7], 0.939)
        self.assertAlmostEqual(motif.counts["A", 8], 0.068)
        self.assertAlmostEqual(motif.counts["T", 0], 0.150)
        self.assertAlmostEqual(motif.counts["T", 1], 0.001)
        self.assertAlmostEqual(motif.counts["T", 2], 0.314)
        self.assertAlmostEqual(motif.counts["T", 3], 0.150)
        self.assertAlmostEqual(motif.counts["T", 4], 0.001)
        self.assertAlmostEqual(motif.counts["T", 5], 0.314)
        self.assertAlmostEqual(motif.counts["T", 6], 0.143)
        self.assertAlmostEqual(motif.counts["T", 7], 0.009)
        self.assertAlmostEqual(motif.counts["T", 8], 0.609)
        self.assertAlmostEqual(motif.counts["C", 0], 0.718)
        self.assertAlmostEqual(motif.counts["C", 1], 0.006)
        self.assertAlmostEqual(motif.counts["C", 2], 0.214)
        self.assertAlmostEqual(motif.counts["C", 3], 0.718)
        self.assertAlmostEqual(motif.counts["C", 4], 0.006)
        self.assertAlmostEqual(motif.counts["C", 5], 0.214)
        self.assertAlmostEqual(motif.counts["C", 6], 0.143)
        self.assertAlmostEqual(motif.counts["C", 7], 0.006)
        self.assertAlmostEqual(motif.counts["C", 8], 0.107)
        self.assertEqual(motif.consensus, "CAACAAAAT")
        self.assertEqual(motif.degenerate_consensus, "CAWCAWAAT")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.79033346,
                        1.79461597,
                        0.33472715,
                        0.79033346,
                        1.79461597,
                        0.33472715,
                        0.60049374,
                        1.60901246,
                        0.47798759,
                    ]
                ),
            )
        )
        self.assertEqual(motif[:-2].consensus, "CAACAAA")

    def test_sites_parsing(self):
        """Test if Bio.motifs can parse JASPAR-style sites files."""
        with open("motifs/Arnt.sites") as stream:
            m = motifs.read(stream, "sites")
        self.assertEqual(m.length, 6)
        self.assertEqual(m.alignment.sequences[0], "CACGTG")
        self.assertEqual(m.alignment.sequences[1], "CACGTG")
        self.assertEqual(m.alignment.sequences[2], "CACGTG")
        self.assertEqual(m.alignment.sequences[3], "CACGTG")
        self.assertEqual(m.alignment.sequences[4], "CACGTG")
        self.assertEqual(m.alignment.sequences[5], "CACGTG")
        self.assertEqual(m.alignment.sequences[6], "CACGTG")
        self.assertEqual(m.alignment.sequences[7], "CACGTG")
        self.assertEqual(m.alignment.sequences[8], "CACGTG")
        self.assertEqual(m.alignment.sequences[9], "CACGTG")
        self.assertEqual(m.alignment.sequences[10], "CACGTG")
        self.assertEqual(m.alignment.sequences[11], "CACGTG")
        self.assertEqual(m.alignment.sequences[12], "CACGTG")
        self.assertEqual(m.alignment.sequences[13], "CACGTG")
        self.assertEqual(m.alignment.sequences[14], "CACGTG")
        self.assertEqual(m.alignment.sequences[15], "AACGTG")
        self.assertEqual(m.alignment.sequences[16], "AACGTG")
        self.assertEqual(m.alignment.sequences[17], "AACGTG")
        self.assertEqual(m.alignment.sequences[18], "AACGTG")
        self.assertEqual(m.alignment.sequences[19], "CGCGTG")
        self.assertAlmostEqual(m.counts["A", 0], 4)
        self.assertAlmostEqual(m.counts["A", 1], 19)
        self.assertAlmostEqual(m.counts["A", 2], 0)
        self.assertAlmostEqual(m.counts["A", 3], 0)
        self.assertAlmostEqual(m.counts["A", 4], 0)
        self.assertAlmostEqual(m.counts["A", 5], 0)
        self.assertAlmostEqual(m.counts["C", 0], 16)
        self.assertAlmostEqual(m.counts["C", 1], 0)
        self.assertAlmostEqual(m.counts["C", 2], 20)
        self.assertAlmostEqual(m.counts["C", 3], 0)
        self.assertAlmostEqual(m.counts["C", 4], 0)
        self.assertAlmostEqual(m.counts["C", 5], 0)
        self.assertAlmostEqual(m.counts["G", 0], 0)
        self.assertAlmostEqual(m.counts["G", 1], 1)
        self.assertAlmostEqual(m.counts["G", 2], 0)
        self.assertAlmostEqual(m.counts["G", 3], 20)
        self.assertAlmostEqual(m.counts["G", 4], 0)
        self.assertAlmostEqual(m.counts["G", 5], 20)
        self.assertAlmostEqual(m.counts["T", 0], 0)
        self.assertAlmostEqual(m.counts["T", 1], 0)
        self.assertAlmostEqual(m.counts["T", 2], 0)
        self.assertAlmostEqual(m.counts["T", 3], 0)
        self.assertAlmostEqual(m.counts["T", 4], 20)
        self.assertAlmostEqual(m.counts["T", 5], 0)
        self.assertEqual(m.consensus, "CACGTG")
        self.assertEqual(m.degenerate_consensus, "CACGTG")
        self.assertTrue(
            np.allclose(
                m.relative_entropy,
                np.array([1.278071905112638, 1.7136030428840439, 2.0, 2.0, 2.0, 2.0]),
            )
        )
        self.assertEqual(m[::2].consensus, "CCT")


class TestMEME(unittest.TestCase):

    def test_meme_parser_1(self):
        """Parse motifs/meme.INO_up800.classic.oops.xml file."""
        with open("motifs/meme.INO_up800.classic.oops.xml") as stream:
            record = motifs.parse(stream, "meme")
        self.assertEqual(record.version, "5.0.1")
        self.assertEqual(record.datafile, "common/INO_up800.s")
        self.assertEqual(record.alphabet, "ACGT")
        self.assertEqual(len(record.sequences), 7)
        self.assertEqual(record.sequences[0], "sequence_0")
        self.assertEqual(record.sequences[1], "sequence_1")
        self.assertEqual(record.sequences[2], "sequence_2")
        self.assertEqual(record.sequences[3], "sequence_3")
        self.assertEqual(record.sequences[4], "sequence_4")
        self.assertEqual(record.sequences[5], "sequence_5")
        self.assertEqual(record.sequences[6], "sequence_6")
        self.assertEqual(
            record.command,
            "meme common/INO_up800.s -oc results/meme10 -mod oops -dna -revcomp -bfile common/yeast.nc.6.freq -nmotifs 2 -objfun classic -minw 8 -nostatus ",
        )
        self.assertEqual(len(record), 2)
        motif = record[0]
        self.assertEqual(motif.name, "GSKGCATGTGAAA")
        self.assertEqual(record["GSKGCATGTGAAA"], motif)
        self.assertEqual(motif.num_occurrences, 7)
        self.assertAlmostEqual(motif.evalue, 0.19)
        self.assertEqual(motif.alphabet, "ACGT")
        self.assertEqual(len(motif.alignment.sequences), 7)
        self.assertAlmostEqual(motif.alignment.sequences[0].pvalue, 1.21e-08, places=10)
        self.assertAlmostEqual(motif.alignment.sequences[1].pvalue, 1.87e-08, places=10)
        self.assertAlmostEqual(motif.alignment.sequences[2].pvalue, 6.62e-08, places=10)
        self.assertAlmostEqual(motif.alignment.sequences[3].pvalue, 1.05e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[4].pvalue, 1.69e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[5].pvalue, 5.62e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[6].pvalue, 1.08e-06, places=8)
        self.assertEqual(motif.alignment.sequences[0].sequence_name, "INO1")
        self.assertEqual(motif.alignment.sequences[1].sequence_name, "FAS1")
        self.assertEqual(motif.alignment.sequences[2].sequence_name, "ACC1")
        self.assertEqual(motif.alignment.sequences[3].sequence_name, "CHO2")
        self.assertEqual(motif.alignment.sequences[4].sequence_name, "CHO1")
        self.assertEqual(motif.alignment.sequences[5].sequence_name, "FAS2")
        self.assertEqual(motif.alignment.sequences[6].sequence_name, "OPI3")
        self.assertEqual(motif.alignment.sequences[0].sequence_id, "sequence_5")
        self.assertEqual(motif.alignment.sequences[1].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[2].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[3].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[4].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[5].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[6].sequence_id, "sequence_6")
        self.assertEqual(motif.alignment.sequences[0].strand, "+")
        self.assertEqual(motif.alignment.sequences[1].strand, "-")
        self.assertEqual(motif.alignment.sequences[2].strand, "-")
        self.assertEqual(motif.alignment.sequences[3].strand, "-")
        self.assertEqual(motif.alignment.sequences[4].strand, "-")
        self.assertEqual(motif.alignment.sequences[5].strand, "-")
        self.assertEqual(motif.alignment.sequences[6].strand, "+")
        self.assertEqual(motif.alignment.sequences[0].length, 13)
        self.assertEqual(motif.alignment.sequences[1].length, 13)
        self.assertEqual(motif.alignment.sequences[2].length, 13)
        self.assertEqual(motif.alignment.sequences[3].length, 13)
        self.assertEqual(motif.alignment.sequences[4].length, 13)
        self.assertEqual(motif.alignment.sequences[5].length, 13)
        self.assertEqual(motif.alignment.sequences[6].length, 13)
        self.assertEqual(motif.alignment.sequences[0].start, 620)
        self.assertEqual(motif.alignment.sequences[1].start, 94)
        self.assertEqual(motif.alignment.sequences[2].start, 82)
        self.assertEqual(motif.alignment.sequences[3].start, 353)
        self.assertEqual(motif.alignment.sequences[4].start, 639)
        self.assertEqual(motif.alignment.sequences[5].start, 566)
        self.assertEqual(motif.alignment.sequences[6].start, 585)
        self.assertEqual(motif.alignment.sequences[0], "GCGGCATGTGAAA")
        self.assertEqual(motif.alignment.sequences[1], "GCGGCATGTGAAG")
        self.assertEqual(motif.alignment.sequences[2], "GGGCCATGTGAAG")
        self.assertEqual(motif.alignment.sequences[3], "GCGGCATGAGAAA")
        self.assertEqual(motif.alignment.sequences[4], "GGTCCATGTGAAA")
        self.assertEqual(motif.alignment.sequences[5], "GTAGCATGTGAAA")
        self.assertEqual(motif.alignment.sequences[6], "AGTGCATGTGGAA")
        self.assertEqual(motif.consensus, "GCGGCATGTGAAA")
        self.assertEqual(motif.degenerate_consensus, "GSKGCATGTGAAA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        1.4083272214176723,
                        0.5511843642748154,
                        0.6212165065138244,
                        1.136879431433369,
                        2.0,
                        2.0,
                        2.0,
                        2.0,
                        1.4083272214176723,
                        2.0,
                        1.4083272214176723,
                        2.0,
                        1.136879431433369,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1::2].consensus, "CGAGGA")
        motif = record[1]
        self.assertEqual(motif.name, "TTGACWCYTGCYCWG")
        self.assertEqual(record["TTGACWCYTGCYCWG"], motif)
        self.assertEqual(motif.num_occurrences, 7)
        self.assertAlmostEqual(motif.evalue, 54)
        self.assertEqual(motif.alphabet, "ACGT")
        self.assertEqual(len(motif.alignment.sequences), 7)
        self.assertAlmostEqual(motif.alignment.sequences[0].pvalue, 7.2e-10, places=11)
        self.assertAlmostEqual(motif.alignment.sequences[1].pvalue, 2.56e-08, places=10)
        self.assertAlmostEqual(motif.alignment.sequences[2].pvalue, 1.59e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[3].pvalue, 2.05e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[4].pvalue, 3.85e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[5].pvalue, 5.11e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[6].pvalue, 8.01e-07, places=9)
        self.assertEqual(motif.alignment.sequences[0].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[1].sequence_id, "sequence_6")
        self.assertEqual(motif.alignment.sequences[2].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[3].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[4].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[5].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[6].sequence_id, "sequence_5")
        self.assertEqual(motif.alignment.sequences[0].strand, "+")
        self.assertEqual(motif.alignment.sequences[1].strand, "-")
        self.assertEqual(motif.alignment.sequences[2].strand, "-")
        self.assertEqual(motif.alignment.sequences[3].strand, "+")
        self.assertEqual(motif.alignment.sequences[4].strand, "+")
        self.assertEqual(motif.alignment.sequences[5].strand, "-")
        self.assertEqual(motif.alignment.sequences[6].strand, "+")
        self.assertEqual(motif.alignment.sequences[0].length, 15)
        self.assertEqual(motif.alignment.sequences[1].length, 15)
        self.assertEqual(motif.alignment.sequences[2].length, 15)
        self.assertEqual(motif.alignment.sequences[3].length, 15)
        self.assertEqual(motif.alignment.sequences[4].length, 15)
        self.assertEqual(motif.alignment.sequences[5].length, 15)
        self.assertEqual(motif.alignment.sequences[6].length, 15)
        self.assertEqual(motif.alignment.sequences[0].start, 104)
        self.assertEqual(motif.alignment.sequences[1].start, 566)
        self.assertEqual(motif.alignment.sequences[2].start, 585)
        self.assertEqual(motif.alignment.sequences[3].start, 30)
        self.assertEqual(motif.alignment.sequences[4].start, 54)
        self.assertEqual(motif.alignment.sequences[5].start, 272)
        self.assertEqual(motif.alignment.sequences[6].start, 214)
        self.assertEqual(motif.alignment.sequences[0], "TTGACACCTGCCCAG")
        self.assertEqual(motif.alignment.sequences[1], "TTGACACCTACCCTG")
        self.assertEqual(motif.alignment.sequences[2], "TTGTCTCTTGCTCTG")
        self.assertEqual(motif.alignment.sequences[3], "TTGACACTTGATCAG")
        self.assertEqual(motif.alignment.sequences[4], "TTCACTACTCCCCTG")
        self.assertEqual(motif.alignment.sequences[5], "TTGACAACGGCTGGG")
        self.assertEqual(motif.alignment.sequences[6], "TTCACGCTTGCTACG")
        self.assertEqual(motif.consensus, "TTGACACCTGCTCTG")
        self.assertEqual(motif.degenerate_consensus, "TTGACWCYTGCYCNG")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        2.0,
                        2.0,
                        1.136879431433369,
                        1.4083272214176723,
                        2.0,
                        0.6212165065138244,
                        1.136879431433369,
                        1.0147718639657484,
                        1.4083272214176723,
                        0.8511651457190834,
                        1.4083272214176723,
                        1.0147718639657484,
                        0.8511651457190834,
                        0.15762900682289133,
                        2.0,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1::2].consensus, "TAACGTT")

    def test_meme_parser_2(self):
        """Parsing motifs/meme.adh.classic.oops.xml file."""
        with open("motifs/meme.adh.classic.oops.xml") as stream:
            record = motifs.parse(stream, "meme")
        self.assertEqual(record.version, "5.0.1")
        self.assertEqual(record.datafile, "common/adh.s")
        self.assertEqual(record.alphabet, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(record.sequences), 33)
        self.assertEqual(record.sequences[0], "sequence_0")
        self.assertEqual(record.sequences[1], "sequence_1")
        self.assertEqual(record.sequences[2], "sequence_2")
        self.assertEqual(record.sequences[3], "sequence_3")
        self.assertEqual(record.sequences[4], "sequence_4")
        self.assertEqual(record.sequences[5], "sequence_5")
        self.assertEqual(record.sequences[6], "sequence_6")
        self.assertEqual(record.sequences[7], "sequence_7")
        self.assertEqual(record.sequences[8], "sequence_8")
        self.assertEqual(record.sequences[9], "sequence_9")
        self.assertEqual(record.sequences[10], "sequence_10")
        self.assertEqual(record.sequences[11], "sequence_11")
        self.assertEqual(record.sequences[12], "sequence_12")
        self.assertEqual(record.sequences[13], "sequence_13")
        self.assertEqual(record.sequences[14], "sequence_14")
        self.assertEqual(record.sequences[15], "sequence_15")
        self.assertEqual(record.sequences[16], "sequence_16")
        self.assertEqual(record.sequences[17], "sequence_17")
        self.assertEqual(record.sequences[18], "sequence_18")
        self.assertEqual(record.sequences[19], "sequence_19")
        self.assertEqual(record.sequences[20], "sequence_20")
        self.assertEqual(record.sequences[21], "sequence_21")
        self.assertEqual(record.sequences[22], "sequence_22")
        self.assertEqual(record.sequences[23], "sequence_23")
        self.assertEqual(record.sequences[24], "sequence_24")
        self.assertEqual(record.sequences[25], "sequence_25")
        self.assertEqual(record.sequences[26], "sequence_26")
        self.assertEqual(record.sequences[27], "sequence_27")
        self.assertEqual(record.sequences[28], "sequence_28")
        self.assertEqual(record.sequences[29], "sequence_29")
        self.assertEqual(record.sequences[30], "sequence_30")
        self.assertEqual(record.sequences[31], "sequence_31")
        self.assertEqual(record.sequences[32], "sequence_32")
        self.assertEqual(
            record.command,
            "meme common/adh.s -oc results/meme4 -mod oops -protein -nmotifs 2 -objfun classic -minw 8 -nostatus ",
        )
        self.assertEqual(len(record), 2)
        motif = record[0]
        self.assertEqual(motif.id, "motif_1")
        self.assertEqual(motif.name, "GKVALVTGAASGJGKATAKAL")
        self.assertEqual(motif.alt_id, "MEME-1")
        self.assertEqual(record["GKVALVTGAASGJGKATAKAL"], motif)
        self.assertEqual(motif.num_occurrences, 33)
        self.assertAlmostEqual(motif.evalue, 4.0e-129, places=130)
        self.assertEqual(motif.alphabet, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(motif.alignment.sequences), 33)
        self.assertAlmostEqual(motif.alignment.sequences[0].pvalue, 8.78e-18, places=20)
        self.assertAlmostEqual(motif.alignment.sequences[1].pvalue, 1.41e-17, places=19)
        self.assertAlmostEqual(motif.alignment.sequences[2].pvalue, 1.42e-16, places=18)
        self.assertAlmostEqual(motif.alignment.sequences[3].pvalue, 2.75e-16, places=18)
        self.assertAlmostEqual(motif.alignment.sequences[4].pvalue, 3.55e-16, places=18)
        self.assertAlmostEqual(motif.alignment.sequences[5].pvalue, 3.55e-16, places=18)
        self.assertAlmostEqual(motif.alignment.sequences[6].pvalue, 1.74e-15, places=17)
        self.assertAlmostEqual(motif.alignment.sequences[7].pvalue, 3.87e-15, places=17)
        self.assertAlmostEqual(motif.alignment.sequences[8].pvalue, 4.84e-15, places=17)
        self.assertAlmostEqual(motif.alignment.sequences[9].pvalue, 1.04e-14, places=16)
        self.assertAlmostEqual(
            motif.alignment.sequences[10].pvalue, 1.58e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[11].pvalue, 1.76e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[12].pvalue, 2.16e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[13].pvalue, 2.94e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[14].pvalue, 3.25e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[15].pvalue, 3.98e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[16].pvalue, 4.39e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[17].pvalue, 4.39e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[18].pvalue, 4.85e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[19].pvalue, 6.52e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[20].pvalue, 1.41e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[21].pvalue, 1.55e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[22].pvalue, 3.07e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[23].pvalue, 5.43e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[24].pvalue, 6.91e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[25].pvalue, 8.76e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[26].pvalue, 9.48e-12, places=14
        )
        self.assertAlmostEqual(motif.alignment.sequences[27].pvalue, 1.2e-11, places=12)
        self.assertAlmostEqual(
            motif.alignment.sequences[28].pvalue, 1.19e-09, places=11
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[29].pvalue, 1.54e-09, places=11
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[30].pvalue, 1.99e-09, places=11
        )
        self.assertAlmostEqual(motif.alignment.sequences[31].pvalue, 1.42e-06, places=8)
        self.assertAlmostEqual(motif.alignment.sequences[32].pvalue, 3.43e-06, places=8)
        self.assertEqual(motif.alignment.sequences[0].sequence_name, "BUDC_KLETE")
        self.assertEqual(motif.alignment.sequences[1].sequence_name, "YINL_LISMO")
        self.assertEqual(motif.alignment.sequences[2].sequence_name, "DHII_HUMAN")
        self.assertEqual(motif.alignment.sequences[3].sequence_name, "HDE_CANTR")
        self.assertEqual(motif.alignment.sequences[4].sequence_name, "YRTP_BACSU")
        self.assertEqual(motif.alignment.sequences[5].sequence_name, "ENTA_ECOLI")
        self.assertEqual(motif.alignment.sequences[6].sequence_name, "HDHA_ECOLI")
        self.assertEqual(motif.alignment.sequences[7].sequence_name, "RIDH_KLEAE")
        self.assertEqual(motif.alignment.sequences[8].sequence_name, "DHB2_HUMAN")
        self.assertEqual(motif.alignment.sequences[9].sequence_name, "FIXR_BRAJA")
        self.assertEqual(motif.alignment.sequences[10].sequence_name, "PCR_PEA")
        self.assertEqual(motif.alignment.sequences[11].sequence_name, "DHCA_HUMAN")
        self.assertEqual(motif.alignment.sequences[12].sequence_name, "BDH_HUMAN")
        self.assertEqual(motif.alignment.sequences[13].sequence_name, "3BHD_COMTE")
        self.assertEqual(motif.alignment.sequences[14].sequence_name, "DHGB_BACME")
        self.assertEqual(motif.alignment.sequences[15].sequence_name, "DHMA_FLAS1")
        self.assertEqual(motif.alignment.sequences[16].sequence_name, "FVT1_HUMAN")
        self.assertEqual(motif.alignment.sequences[17].sequence_name, "BA72_EUBSP")
        self.assertEqual(motif.alignment.sequences[18].sequence_name, "BPHB_PSEPS")
        self.assertEqual(motif.alignment.sequences[19].sequence_name, "DHB3_HUMAN")
        self.assertEqual(motif.alignment.sequences[20].sequence_name, "DHES_HUMAN")
        self.assertEqual(motif.alignment.sequences[21].sequence_name, "AP27_MOUSE")
        self.assertEqual(motif.alignment.sequences[22].sequence_name, "2BHD_STREX")
        self.assertEqual(motif.alignment.sequences[23].sequence_name, "NODG_RHIME")
        self.assertEqual(motif.alignment.sequences[24].sequence_name, "HMTR_LEIMA")
        self.assertEqual(motif.alignment.sequences[25].sequence_name, "LIGD_PSEPA")
        self.assertEqual(motif.alignment.sequences[26].sequence_name, "MAS1_AGRRA")
        self.assertEqual(motif.alignment.sequences[27].sequence_name, "RFBB_NEIGO")
        self.assertEqual(motif.alignment.sequences[28].sequence_name, "GUTD_ECOLI")
        self.assertEqual(motif.alignment.sequences[29].sequence_name, "ADH_DROME")
        self.assertEqual(motif.alignment.sequences[30].sequence_name, "FABI_ECOLI")
        self.assertEqual(motif.alignment.sequences[31].sequence_name, "CSGA_MYXXA")
        self.assertEqual(motif.alignment.sequences[32].sequence_name, "YURA_MYXXA")
        self.assertEqual(motif.alignment.sequences[0].sequence_id, "sequence_7")
        self.assertEqual(motif.alignment.sequences[1].sequence_id, "sequence_20")
        self.assertEqual(motif.alignment.sequences[2].sequence_id, "sequence_10")
        self.assertEqual(motif.alignment.sequences[3].sequence_id, "sequence_15")
        self.assertEqual(motif.alignment.sequences[4].sequence_id, "sequence_21")
        self.assertEqual(motif.alignment.sequences[5].sequence_id, "sequence_12")
        self.assertEqual(motif.alignment.sequences[6].sequence_id, "sequence_16")
        self.assertEqual(motif.alignment.sequences[7].sequence_id, "sequence_19")
        self.assertEqual(motif.alignment.sequences[8].sequence_id, "sequence_23")
        self.assertEqual(motif.alignment.sequences[9].sequence_id, "sequence_13")
        self.assertEqual(motif.alignment.sequences[10].sequence_id, "sequence_30")
        self.assertEqual(motif.alignment.sequences[11].sequence_id, "sequence_25")
        self.assertEqual(motif.alignment.sequences[12].sequence_id, "sequence_5")
        self.assertEqual(motif.alignment.sequences[13].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[14].sequence_id, "sequence_9")
        self.assertEqual(motif.alignment.sequences[15].sequence_id, "sequence_11")
        self.assertEqual(motif.alignment.sequences[16].sequence_id, "sequence_27")
        self.assertEqual(motif.alignment.sequences[17].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[18].sequence_id, "sequence_6")
        self.assertEqual(motif.alignment.sequences[19].sequence_id, "sequence_24")
        self.assertEqual(motif.alignment.sequences[20].sequence_id, "sequence_8")
        self.assertEqual(motif.alignment.sequences[21].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[22].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[23].sequence_id, "sequence_18")
        self.assertEqual(motif.alignment.sequences[24].sequence_id, "sequence_28")
        self.assertEqual(motif.alignment.sequences[25].sequence_id, "sequence_17")
        self.assertEqual(motif.alignment.sequences[26].sequence_id, "sequence_29")
        self.assertEqual(motif.alignment.sequences[27].sequence_id, "sequence_31")
        self.assertEqual(motif.alignment.sequences[28].sequence_id, "sequence_14")
        self.assertEqual(motif.alignment.sequences[29].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[30].sequence_id, "sequence_26")
        self.assertEqual(motif.alignment.sequences[31].sequence_id, "sequence_22")
        self.assertEqual(motif.alignment.sequences[32].sequence_id, "sequence_32")
        self.assertEqual(motif.alignment.sequences[0].strand, "+")
        self.assertEqual(motif.alignment.sequences[1].strand, "+")
        self.assertEqual(motif.alignment.sequences[2].strand, "+")
        self.assertEqual(motif.alignment.sequences[3].strand, "+")
        self.assertEqual(motif.alignment.sequences[4].strand, "+")
        self.assertEqual(motif.alignment.sequences[5].strand, "+")
        self.assertEqual(motif.alignment.sequences[6].strand, "+")
        self.assertEqual(motif.alignment.sequences[7].strand, "+")
        self.assertEqual(motif.alignment.sequences[8].strand, "+")
        self.assertEqual(motif.alignment.sequences[9].strand, "+")
        self.assertEqual(motif.alignment.sequences[10].strand, "+")
        self.assertEqual(motif.alignment.sequences[11].strand, "+")
        self.assertEqual(motif.alignment.sequences[12].strand, "+")
        self.assertEqual(motif.alignment.sequences[13].strand, "+")
        self.assertEqual(motif.alignment.sequences[14].strand, "+")
        self.assertEqual(motif.alignment.sequences[15].strand, "+")
        self.assertEqual(motif.alignment.sequences[16].strand, "+")
        self.assertEqual(motif.alignment.sequences[17].strand, "+")
        self.assertEqual(motif.alignment.sequences[18].strand, "+")
        self.assertEqual(motif.alignment.sequences[19].strand, "+")
        self.assertEqual(motif.alignment.sequences[20].strand, "+")
        self.assertEqual(motif.alignment.sequences[21].strand, "+")
        self.assertEqual(motif.alignment.sequences[22].strand, "+")
        self.assertEqual(motif.alignment.sequences[23].strand, "+")
        self.assertEqual(motif.alignment.sequences[24].strand, "+")
        self.assertEqual(motif.alignment.sequences[25].strand, "+")
        self.assertEqual(motif.alignment.sequences[26].strand, "+")
        self.assertEqual(motif.alignment.sequences[27].strand, "+")
        self.assertEqual(motif.alignment.sequences[28].strand, "+")
        self.assertEqual(motif.alignment.sequences[29].strand, "+")
        self.assertEqual(motif.alignment.sequences[30].strand, "+")
        self.assertEqual(motif.alignment.sequences[31].strand, "+")
        self.assertEqual(motif.alignment.sequences[32].strand, "+")
        self.assertEqual(motif.alignment.sequences[0].length, 21)
        self.assertEqual(motif.alignment.sequences[1].length, 21)
        self.assertEqual(motif.alignment.sequences[2].length, 21)
        self.assertEqual(motif.alignment.sequences[3].length, 21)
        self.assertEqual(motif.alignment.sequences[4].length, 21)
        self.assertEqual(motif.alignment.sequences[5].length, 21)
        self.assertEqual(motif.alignment.sequences[6].length, 21)
        self.assertEqual(motif.alignment.sequences[7].length, 21)
        self.assertEqual(motif.alignment.sequences[8].length, 21)
        self.assertEqual(motif.alignment.sequences[9].length, 21)
        self.assertEqual(motif.alignment.sequences[10].length, 21)
        self.assertEqual(motif.alignment.sequences[11].length, 21)
        self.assertEqual(motif.alignment.sequences[12].length, 21)
        self.assertEqual(motif.alignment.sequences[13].length, 21)
        self.assertEqual(motif.alignment.sequences[14].length, 21)
        self.assertEqual(motif.alignment.sequences[15].length, 21)
        self.assertEqual(motif.alignment.sequences[16].length, 21)
        self.assertEqual(motif.alignment.sequences[17].length, 21)
        self.assertEqual(motif.alignment.sequences[18].length, 21)
        self.assertEqual(motif.alignment.sequences[19].length, 21)
        self.assertEqual(motif.alignment.sequences[20].length, 21)
        self.assertEqual(motif.alignment.sequences[21].length, 21)
        self.assertEqual(motif.alignment.sequences[22].length, 21)
        self.assertEqual(motif.alignment.sequences[23].length, 21)
        self.assertEqual(motif.alignment.sequences[24].length, 21)
        self.assertEqual(motif.alignment.sequences[25].length, 21)
        self.assertEqual(motif.alignment.sequences[26].length, 21)
        self.assertEqual(motif.alignment.sequences[27].length, 21)
        self.assertEqual(motif.alignment.sequences[28].length, 21)
        self.assertEqual(motif.alignment.sequences[29].length, 21)
        self.assertEqual(motif.alignment.sequences[30].length, 21)
        self.assertEqual(motif.alignment.sequences[31].length, 21)
        self.assertEqual(motif.alignment.sequences[32].length, 21)
        self.assertEqual(motif.alignment.sequences[0].start, 2)
        self.assertEqual(motif.alignment.sequences[1].start, 5)
        self.assertEqual(motif.alignment.sequences[2].start, 34)
        self.assertEqual(motif.alignment.sequences[3].start, 322)
        self.assertEqual(motif.alignment.sequences[4].start, 6)
        self.assertEqual(motif.alignment.sequences[5].start, 5)
        self.assertEqual(motif.alignment.sequences[6].start, 11)
        self.assertEqual(motif.alignment.sequences[7].start, 14)
        self.assertEqual(motif.alignment.sequences[8].start, 82)
        self.assertEqual(motif.alignment.sequences[9].start, 36)
        self.assertEqual(motif.alignment.sequences[10].start, 86)
        self.assertEqual(motif.alignment.sequences[11].start, 4)
        self.assertEqual(motif.alignment.sequences[12].start, 55)
        self.assertEqual(motif.alignment.sequences[13].start, 6)
        self.assertEqual(motif.alignment.sequences[14].start, 7)
        self.assertEqual(motif.alignment.sequences[15].start, 14)
        self.assertEqual(motif.alignment.sequences[16].start, 32)
        self.assertEqual(motif.alignment.sequences[17].start, 6)
        self.assertEqual(motif.alignment.sequences[18].start, 5)
        self.assertEqual(motif.alignment.sequences[19].start, 48)
        self.assertEqual(motif.alignment.sequences[20].start, 2)
        self.assertEqual(motif.alignment.sequences[21].start, 7)
        self.assertEqual(motif.alignment.sequences[22].start, 6)
        self.assertEqual(motif.alignment.sequences[23].start, 6)
        self.assertEqual(motif.alignment.sequences[24].start, 6)
        self.assertEqual(motif.alignment.sequences[25].start, 6)
        self.assertEqual(motif.alignment.sequences[26].start, 245)
        self.assertEqual(motif.alignment.sequences[27].start, 6)
        self.assertEqual(motif.alignment.sequences[28].start, 2)
        self.assertEqual(motif.alignment.sequences[29].start, 6)
        self.assertEqual(motif.alignment.sequences[30].start, 6)
        self.assertEqual(motif.alignment.sequences[31].start, 13)
        self.assertEqual(motif.alignment.sequences[32].start, 116)
        self.assertEqual(motif.alignment.sequences[0], "QKVALVTGAGQGIGKAIALRL")
        self.assertEqual(motif.alignment.sequences[1], "NKVIIITGASSGIGKATALLL")
        self.assertEqual(motif.alignment.sequences[2], "GKKVIVTGASKGIGREMAYHL")
        self.assertEqual(motif.alignment.sequences[3], "DKVVLITGAGAGLGKEYAKWF")
        self.assertEqual(motif.alignment.sequences[4], "HKTALITGGGRGIGRATALAL")
        self.assertEqual(motif.alignment.sequences[5], "GKNVWVTGAGKGIGYATALAF")
        self.assertEqual(motif.alignment.sequences[6], "GKCAIITGAGAGIGKEIAITF")
        self.assertEqual(motif.alignment.sequences[7], "GKVAAITGAASGIGLECARTL")
        self.assertEqual(motif.alignment.sequences[8], "QKAVLVTGGDCGLGHALCKYL")
        self.assertEqual(motif.alignment.sequences[9], "PKVMLLTGASRGIGHATAKLF")
        self.assertEqual(motif.alignment.sequences[10], "KGNVVITGASSGLGLATAKAL")
        self.assertEqual(motif.alignment.sequences[11], "IHVALVTGGNKGIGLAIVRDL")
        self.assertEqual(motif.alignment.sequences[12], "SKAVLVTGCDSGFGFSLAKHL")
        self.assertEqual(motif.alignment.sequences[13], "GKVALVTGGASGVGLEVVKLL")
        self.assertEqual(motif.alignment.sequences[14], "GKVVVITGSSTGLGKSMAIRF")
        self.assertEqual(motif.alignment.sequences[15], "GKAAIVTGAAGGIGRATVEAY")
        self.assertEqual(motif.alignment.sequences[16], "GAHVVVTGGSSGIGKCIAIEC")
        self.assertEqual(motif.alignment.sequences[17], "DKVTIITGGTRGIGFAAAKIF")
        self.assertEqual(motif.alignment.sequences[18], "GEAVLITGGASGLGRALVDRF")
        self.assertEqual(motif.alignment.sequences[19], "GQWAVITGAGDGIGKAYSFEL")
        self.assertEqual(motif.alignment.sequences[20], "RTVVLITGCSSGIGLHLAVRL")
        self.assertEqual(motif.alignment.sequences[21], "GLRALVTGAGKGIGRDTVKAL")
        self.assertEqual(motif.alignment.sequences[22], "GKTVIITGGARGLGAEAARQA")
        self.assertEqual(motif.alignment.sequences[23], "GRKALVTGASGAIGGAIARVL")
        self.assertEqual(motif.alignment.sequences[24], "VPVALVTGAAKRLGRSIAEGL")
        self.assertEqual(motif.alignment.sequences[25], "DQVAFITGGASGAGFGQAKVF")
        self.assertEqual(motif.alignment.sequences[26], "SPVILVSGSNRGVGKAIAEDL")
        self.assertEqual(motif.alignment.sequences[27], "KKNILVTGGAGFIGSAVVRHI")
        self.assertEqual(motif.alignment.sequences[28], "NQVAVVIGGGQTLGAFLCHGL")
        self.assertEqual(motif.alignment.sequences[29], "NKNVIFVAGLGGIGLDTSKEL")
        self.assertEqual(motif.alignment.sequences[30], "GKRILVTGVASKLSIAYGIAQ")
        self.assertEqual(motif.alignment.sequences[31], "VDVLINNAGVSGLWCALGDVD")
        self.assertEqual(motif.alignment.sequences[32], "IIDTNVTGAAATLSAVLPQMV")
        self.assertEqual(motif.consensus, "GKVALVTGAASGIGKATAKAL")
        self.assertEqual(motif[2:8].consensus, "VALVTG")
        motif = record[1]
        self.assertEqual(motif.name, "VGNPGASAYSASKAAVRGLTESLALELAP")
        self.assertEqual(motif.alt_id, "MEME-2")
        self.assertEqual(record["VGNPGASAYSASKAAVRGLTESLALELAP"], motif)
        self.assertEqual(motif.num_occurrences, 33)
        self.assertAlmostEqual(motif.evalue, 3.1e-130, places=131)
        self.assertEqual(motif.alphabet, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(motif.alignment.sequences), 33)
        self.assertAlmostEqual(motif.alignment.sequences[0].pvalue, 2.09e-21, places=23)
        self.assertAlmostEqual(motif.alignment.sequences[1].pvalue, 7.63e-20, places=22)
        self.assertAlmostEqual(motif.alignment.sequences[2].pvalue, 6.49e-19, places=21)
        self.assertAlmostEqual(motif.alignment.sequences[3].pvalue, 1.92e-18, places=20)
        self.assertAlmostEqual(motif.alignment.sequences[4].pvalue, 5.46e-18, places=20)
        self.assertAlmostEqual(motif.alignment.sequences[5].pvalue, 6.21e-18, places=20)
        self.assertAlmostEqual(motif.alignment.sequences[6].pvalue, 4.52e-17, places=19)
        self.assertAlmostEqual(motif.alignment.sequences[7].pvalue, 4.52e-17, places=19)
        self.assertAlmostEqual(motif.alignment.sequences[8].pvalue, 9.21e-17, places=19)
        self.assertAlmostEqual(motif.alignment.sequences[9].pvalue, 1.65e-16, places=18)
        self.assertAlmostEqual(
            motif.alignment.sequences[10].pvalue, 2.07e-16, places=18
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[11].pvalue, 3.65e-16, places=18
        )
        self.assertAlmostEqual(motif.alignment.sequences[12].pvalue, 5.7e-16, places=17)
        self.assertAlmostEqual(motif.alignment.sequences[13].pvalue, 5.7e-16, places=17)
        self.assertAlmostEqual(
            motif.alignment.sequences[14].pvalue, 7.93e-16, places=18
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[15].pvalue, 8.85e-16, places=18
        )
        self.assertAlmostEqual(motif.alignment.sequences[16].pvalue, 1.1e-15, places=16)
        self.assertAlmostEqual(
            motif.alignment.sequences[17].pvalue, 1.69e-15, places=17
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[18].pvalue, 3.54e-15, places=17
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[19].pvalue, 4.83e-15, places=17
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[20].pvalue, 7.27e-15, places=17
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[21].pvalue, 9.85e-15, places=17
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[22].pvalue, 2.41e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[23].pvalue, 2.66e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[24].pvalue, 1.22e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[25].pvalue, 5.18e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[26].pvalue, 1.24e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[27].pvalue, 1.35e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[28].pvalue, 5.59e-12, places=14
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[29].pvalue, 1.44e-10, places=12
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[30].pvalue, 1.61e-08, places=10
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[31].pvalue, 4.26e-08, places=10
        )
        self.assertAlmostEqual(motif.alignment.sequences[32].pvalue, 1.16e-07, places=9)
        self.assertEqual(motif.alignment.sequences[0].sequence_name, "BUDC_KLETE")
        self.assertEqual(motif.alignment.sequences[1].sequence_name, "NODG_RHIME")
        self.assertEqual(motif.alignment.sequences[2].sequence_name, "FVT1_HUMAN")
        self.assertEqual(motif.alignment.sequences[3].sequence_name, "DHES_HUMAN")
        self.assertEqual(motif.alignment.sequences[4].sequence_name, "DHB3_HUMAN")
        self.assertEqual(motif.alignment.sequences[5].sequence_name, "YRTP_BACSU")
        self.assertEqual(motif.alignment.sequences[6].sequence_name, "HMTR_LEIMA")
        self.assertEqual(motif.alignment.sequences[7].sequence_name, "HDE_CANTR")
        self.assertEqual(motif.alignment.sequences[8].sequence_name, "DHGB_BACME")
        self.assertEqual(motif.alignment.sequences[9].sequence_name, "GUTD_ECOLI")
        self.assertEqual(motif.alignment.sequences[10].sequence_name, "3BHD_COMTE")
        self.assertEqual(motif.alignment.sequences[11].sequence_name, "DHII_HUMAN")
        self.assertEqual(motif.alignment.sequences[12].sequence_name, "BPHB_PSEPS")
        self.assertEqual(motif.alignment.sequences[13].sequence_name, "AP27_MOUSE")
        self.assertEqual(motif.alignment.sequences[14].sequence_name, "BDH_HUMAN")
        self.assertEqual(motif.alignment.sequences[15].sequence_name, "YINL_LISMO")
        self.assertEqual(motif.alignment.sequences[16].sequence_name, "FIXR_BRAJA")
        self.assertEqual(motif.alignment.sequences[17].sequence_name, "2BHD_STREX")
        self.assertEqual(motif.alignment.sequences[18].sequence_name, "RFBB_NEIGO")
        self.assertEqual(motif.alignment.sequences[19].sequence_name, "YURA_MYXXA")
        self.assertEqual(motif.alignment.sequences[20].sequence_name, "RIDH_KLEAE")
        self.assertEqual(motif.alignment.sequences[21].sequence_name, "DHMA_FLAS1")
        self.assertEqual(motif.alignment.sequences[22].sequence_name, "DHB2_HUMAN")
        self.assertEqual(motif.alignment.sequences[23].sequence_name, "HDHA_ECOLI")
        self.assertEqual(motif.alignment.sequences[24].sequence_name, "ENTA_ECOLI")
        self.assertEqual(motif.alignment.sequences[25].sequence_name, "LIGD_PSEPA")
        self.assertEqual(motif.alignment.sequences[26].sequence_name, "CSGA_MYXXA")
        self.assertEqual(motif.alignment.sequences[27].sequence_name, "BA72_EUBSP")
        self.assertEqual(motif.alignment.sequences[28].sequence_name, "ADH_DROME")
        self.assertEqual(motif.alignment.sequences[29].sequence_name, "MAS1_AGRRA")
        self.assertEqual(motif.alignment.sequences[30].sequence_name, "PCR_PEA")
        self.assertEqual(motif.alignment.sequences[31].sequence_name, "FABI_ECOLI")
        self.assertEqual(motif.alignment.sequences[32].sequence_name, "DHCA_HUMAN")
        self.assertEqual(motif.alignment.sequences[0].sequence_id, "sequence_7")
        self.assertEqual(motif.alignment.sequences[1].sequence_id, "sequence_18")
        self.assertEqual(motif.alignment.sequences[2].sequence_id, "sequence_27")
        self.assertEqual(motif.alignment.sequences[3].sequence_id, "sequence_8")
        self.assertEqual(motif.alignment.sequences[4].sequence_id, "sequence_24")
        self.assertEqual(motif.alignment.sequences[5].sequence_id, "sequence_21")
        self.assertEqual(motif.alignment.sequences[6].sequence_id, "sequence_28")
        self.assertEqual(motif.alignment.sequences[7].sequence_id, "sequence_15")
        self.assertEqual(motif.alignment.sequences[8].sequence_id, "sequence_9")
        self.assertEqual(motif.alignment.sequences[9].sequence_id, "sequence_14")
        self.assertEqual(motif.alignment.sequences[10].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[11].sequence_id, "sequence_10")
        self.assertEqual(motif.alignment.sequences[12].sequence_id, "sequence_6")
        self.assertEqual(motif.alignment.sequences[13].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[14].sequence_id, "sequence_5")
        self.assertEqual(motif.alignment.sequences[15].sequence_id, "sequence_20")
        self.assertEqual(motif.alignment.sequences[16].sequence_id, "sequence_13")
        self.assertEqual(motif.alignment.sequences[17].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[18].sequence_id, "sequence_31")
        self.assertEqual(motif.alignment.sequences[19].sequence_id, "sequence_32")
        self.assertEqual(motif.alignment.sequences[20].sequence_id, "sequence_19")
        self.assertEqual(motif.alignment.sequences[21].sequence_id, "sequence_11")
        self.assertEqual(motif.alignment.sequences[22].sequence_id, "sequence_23")
        self.assertEqual(motif.alignment.sequences[23].sequence_id, "sequence_16")
        self.assertEqual(motif.alignment.sequences[24].sequence_id, "sequence_12")
        self.assertEqual(motif.alignment.sequences[25].sequence_id, "sequence_17")
        self.assertEqual(motif.alignment.sequences[26].sequence_id, "sequence_22")
        self.assertEqual(motif.alignment.sequences[27].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[28].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[29].sequence_id, "sequence_29")
        self.assertEqual(motif.alignment.sequences[30].sequence_id, "sequence_30")
        self.assertEqual(motif.alignment.sequences[31].sequence_id, "sequence_26")
        self.assertEqual(motif.alignment.sequences[32].sequence_id, "sequence_25")
        self.assertEqual(motif.alignment.sequences[0].start, 144)
        self.assertEqual(motif.alignment.sequences[1].start, 144)
        self.assertEqual(motif.alignment.sequences[2].start, 178)
        self.assertEqual(motif.alignment.sequences[3].start, 147)
        self.assertEqual(motif.alignment.sequences[4].start, 190)
        self.assertEqual(motif.alignment.sequences[5].start, 147)
        self.assertEqual(motif.alignment.sequences[6].start, 185)
        self.assertEqual(motif.alignment.sequences[7].start, 459)
        self.assertEqual(motif.alignment.sequences[8].start, 152)
        self.assertEqual(motif.alignment.sequences[9].start, 146)
        self.assertEqual(motif.alignment.sequences[10].start, 143)
        self.assertEqual(motif.alignment.sequences[11].start, 175)
        self.assertEqual(motif.alignment.sequences[12].start, 145)
        self.assertEqual(motif.alignment.sequences[13].start, 141)
        self.assertEqual(motif.alignment.sequences[14].start, 200)
        self.assertEqual(motif.alignment.sequences[15].start, 146)
        self.assertEqual(motif.alignment.sequences[16].start, 181)
        self.assertEqual(motif.alignment.sequences[17].start, 144)
        self.assertEqual(motif.alignment.sequences[18].start, 157)
        self.assertEqual(motif.alignment.sequences[19].start, 152)
        self.assertEqual(motif.alignment.sequences[20].start, 152)
        self.assertEqual(motif.alignment.sequences[21].start, 157)
        self.assertEqual(motif.alignment.sequences[22].start, 224)
        self.assertEqual(motif.alignment.sequences[23].start, 151)
        self.assertEqual(motif.alignment.sequences[24].start, 136)
        self.assertEqual(motif.alignment.sequences[25].start, 149)
        self.assertEqual(motif.alignment.sequences[26].start, 80)
        self.assertEqual(motif.alignment.sequences[27].start, 149)
        self.assertEqual(motif.alignment.sequences[28].start, 144)
        self.assertEqual(motif.alignment.sequences[29].start, 384)
        self.assertEqual(motif.alignment.sequences[30].start, 18)
        self.assertEqual(motif.alignment.sequences[31].start, 177)
        self.assertEqual(motif.alignment.sequences[32].start, 144)
        self.assertEqual(motif.alignment.sequences[0], "VGNPELAVYSSSKFAVRGLTQTAARDLAP")
        self.assertEqual(motif.alignment.sequences[1], "IGNPGQTNYCASKAGMIGFSKSLAQEIAT")
        self.assertEqual(motif.alignment.sequences[2], "LGLFGFTAYSASKFAIRGLAEALQMEVKP")
        self.assertEqual(motif.alignment.sequences[3], "MGLPFNDVYCASKFALEGLCESLAVLLLP")
        self.assertEqual(motif.alignment.sequences[4], "FPWPLYSMYSASKAFVCAFSKALQEEYKA")
        self.assertEqual(motif.alignment.sequences[5], "RGAAVTSAYSASKFAVLGLTESLMQEVRK")
        self.assertEqual(motif.alignment.sequences[6], "QPLLGYTIYTMAKGALEGLTRSAALELAP")
        self.assertEqual(motif.alignment.sequences[7], "YGNFGQANYSSSKAGILGLSKTMAIEGAK")
        self.assertEqual(motif.alignment.sequences[8], "IPWPLFVHYAASKGGMKLMTETLALEYAP")
        self.assertEqual(motif.alignment.sequences[9], "VGSKHNSGYSAAKFGGVGLTQSLALDLAE")
        self.assertEqual(motif.alignment.sequences[10], "LPIEQYAGYSASKAAVSALTRAAALSCRK")
        self.assertEqual(motif.alignment.sequences[11], "VAYPMVAAYSASKFALDGFFSSIRKEYSV")
        self.assertEqual(motif.alignment.sequences[12], "YPNGGGPLYTAAKQAIVGLVRELAFELAP")
        self.assertEqual(motif.alignment.sequences[13], "VTFPNLITYSSTKGAMTMLTKAMAMELGP")
        self.assertEqual(motif.alignment.sequences[14], "MANPARSPYCITKFGVEAFSDCLRYEMYP")
        self.assertEqual(motif.alignment.sequences[15], "KAYPGGAVYGATKWAVRDLMEVLRMESAQ")
        self.assertEqual(motif.alignment.sequences[16], "VHPFAGSAYATSKAALASLTRELAHDYAP")
        self.assertEqual(motif.alignment.sequences[17], "MGLALTSSYGASKWGVRGLSKLAAVELGT")
        self.assertEqual(motif.alignment.sequences[18], "TPYAPSSPYSASKAAADHLVRAWQRTYRL")
        self.assertEqual(motif.alignment.sequences[19], "FRGLPATRYSASKAFLSTFMESLRVDLRG")
        self.assertEqual(motif.alignment.sequences[20], "VPVIWEPVYTASKFAVQAFVHTTRRQVAQ")
        self.assertEqual(motif.alignment.sequences[21], "MAEPEAAAYVAAKGGVAMLTRAMAVDLAR")
        self.assertEqual(motif.alignment.sequences[22], "APMERLASYGSSKAAVTMFSSVMRLELSK")
        self.assertEqual(motif.alignment.sequences[23], "NKNINMTSYASSKAAASHLVRNMAFDLGE")
        self.assertEqual(motif.alignment.sequences[24], "TPRIGMSAYGASKAALKSLALSVGLELAG")
        self.assertEqual(motif.alignment.sequences[25], "MGSALAGPYSAAKAASINLMEGYRQGLEK")
        self.assertEqual(motif.alignment.sequences[26], "NTDGGAYAYRMSKAALNMAVRSMSTDLRP")
        self.assertEqual(motif.alignment.sequences[27], "FGSLSGVGYPASKASVIGLTHGLGREIIR")
        self.assertEqual(motif.alignment.sequences[28], "NAIYQVPVYSGTKAAVVNFTSSLAKLAPI")
        self.assertEqual(motif.alignment.sequences[29], "RVLNPLVGYNMTKHALGGLTKTTQHVGWD")
        self.assertEqual(motif.alignment.sequences[30], "EGKIGASLKDSTLFGVSSLSDSLKGDFTS")
        self.assertEqual(motif.alignment.sequences[31], "MGPEGVRVNAISAGPIRTLAASGIKDFRK")
        self.assertEqual(motif.alignment.sequences[32], "RALKSCSPELQQKFRSETITEEELVGLMN")
        self.assertEqual(motif.consensus, "MGLPGASAYSASKAAVRGLTESLALELAP")
        self.assertEqual(motif[-8:-2].consensus, "SLALEL")

    def test_meme_parser_3(self):
        """Parse motifs/meme.farntrans5.classic.anr.xml file."""
        with open("motifs/meme.farntrans5.classic.anr.xml") as stream:
            record = motifs.parse(stream, "meme")
        self.assertEqual(record.version, "5.0.1")
        self.assertEqual(record.datafile, "common/farntrans5.s")
        self.assertEqual(record.alphabet, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(record.sequences), 5)
        self.assertEqual(record.sequences[0], "sequence_0")
        self.assertEqual(record.sequences[1], "sequence_1")
        self.assertEqual(record.sequences[2], "sequence_2")
        self.assertEqual(record.sequences[3], "sequence_3")
        self.assertEqual(record.sequences[4], "sequence_4")
        self.assertEqual(
            record.command,
            "meme common/farntrans5.s -oc results/meme15 -mod anr -protein -nmotifs 2 -objfun classic -minw 8 -nostatus ",
        )
        self.assertEqual(len(record), 2)
        motif = record[0]
        self.assertEqual(motif.name, "GGFGGRPGKEVDLCYTYCALAALAJLGSLD")
        self.assertEqual(record["GGFGGRPGKEVDLCYTYCALAALAJLGSLD"], motif)
        self.assertEqual(motif.num_occurrences, 24)
        self.assertAlmostEqual(motif.evalue, 2.2e-94, places=95)
        self.assertEqual(motif.alphabet, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(motif.alignment.sequences), 24)
        self.assertAlmostEqual(motif.alignment.sequences[0].pvalue, 6.98e-22, places=24)
        self.assertAlmostEqual(motif.alignment.sequences[1].pvalue, 4.67e-21, places=23)
        self.assertAlmostEqual(motif.alignment.sequences[2].pvalue, 1.25e-19, places=21)
        self.assertAlmostEqual(motif.alignment.sequences[3].pvalue, 1.56e-19, places=21)
        self.assertAlmostEqual(motif.alignment.sequences[4].pvalue, 2.44e-19, places=21)
        self.assertAlmostEqual(motif.alignment.sequences[5].pvalue, 6.47e-19, places=21)
        self.assertAlmostEqual(motif.alignment.sequences[6].pvalue, 8.9e-19, places=20)
        self.assertAlmostEqual(motif.alignment.sequences[7].pvalue, 2.53e-18, places=20)
        self.assertAlmostEqual(motif.alignment.sequences[8].pvalue, 1.27e-17, places=19)
        self.assertAlmostEqual(motif.alignment.sequences[9].pvalue, 2.77e-17, places=19)
        self.assertAlmostEqual(
            motif.alignment.sequences[10].pvalue, 4.93e-17, places=19
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[11].pvalue, 7.19e-17, places=19
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[12].pvalue, 8.68e-17, places=19
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[13].pvalue, 2.62e-16, places=18
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[14].pvalue, 2.87e-16, places=18
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[15].pvalue, 7.66e-15, places=17
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[16].pvalue, 2.21e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[17].pvalue, 3.29e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[18].pvalue, 7.21e-14, places=16
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[19].pvalue, 1.14e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[20].pvalue, 1.67e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[21].pvalue, 4.42e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[22].pvalue, 5.11e-13, places=15
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[23].pvalue, 2.82e-10, places=12
        )
        self.assertEqual(motif.alignment.sequences[0].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[1].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[2].sequence_name, "CAL1_YEAST")
        self.assertEqual(motif.alignment.sequences[3].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[4].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[5].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[6].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[7].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[8].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[9].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[10].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[11].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[12].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[13].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[14].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[15].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[16].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[17].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[18].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[19].sequence_name, "CAL1_YEAST")
        self.assertEqual(motif.alignment.sequences[20].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[21].sequence_name, "CAL1_YEAST")
        self.assertEqual(motif.alignment.sequences[22].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[23].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[0].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[1].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[2].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[3].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[4].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[5].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[6].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[7].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[8].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[9].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[10].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[11].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[12].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[13].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[14].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[15].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[16].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[17].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[18].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[19].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[20].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[21].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[22].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[23].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[0].strand, "+")
        self.assertEqual(motif.alignment.sequences[1].strand, "+")
        self.assertEqual(motif.alignment.sequences[2].strand, "+")
        self.assertEqual(motif.alignment.sequences[3].strand, "+")
        self.assertEqual(motif.alignment.sequences[4].strand, "+")
        self.assertEqual(motif.alignment.sequences[5].strand, "+")
        self.assertEqual(motif.alignment.sequences[6].strand, "+")
        self.assertEqual(motif.alignment.sequences[7].strand, "+")
        self.assertEqual(motif.alignment.sequences[8].strand, "+")
        self.assertEqual(motif.alignment.sequences[9].strand, "+")
        self.assertEqual(motif.alignment.sequences[10].strand, "+")
        self.assertEqual(motif.alignment.sequences[11].strand, "+")
        self.assertEqual(motif.alignment.sequences[12].strand, "+")
        self.assertEqual(motif.alignment.sequences[13].strand, "+")
        self.assertEqual(motif.alignment.sequences[14].strand, "+")
        self.assertEqual(motif.alignment.sequences[15].strand, "+")
        self.assertEqual(motif.alignment.sequences[16].strand, "+")
        self.assertEqual(motif.alignment.sequences[17].strand, "+")
        self.assertEqual(motif.alignment.sequences[18].strand, "+")
        self.assertEqual(motif.alignment.sequences[19].strand, "+")
        self.assertEqual(motif.alignment.sequences[20].strand, "+")
        self.assertEqual(motif.alignment.sequences[21].strand, "+")
        self.assertEqual(motif.alignment.sequences[22].strand, "+")
        self.assertEqual(motif.alignment.sequences[23].strand, "+")
        self.assertEqual(motif.alignment.sequences[0].length, 30)
        self.assertEqual(motif.alignment.sequences[1].length, 30)
        self.assertEqual(motif.alignment.sequences[2].length, 30)
        self.assertEqual(motif.alignment.sequences[3].length, 30)
        self.assertEqual(motif.alignment.sequences[4].length, 30)
        self.assertEqual(motif.alignment.sequences[5].length, 30)
        self.assertEqual(motif.alignment.sequences[6].length, 30)
        self.assertEqual(motif.alignment.sequences[7].length, 30)
        self.assertEqual(motif.alignment.sequences[8].length, 30)
        self.assertEqual(motif.alignment.sequences[9].length, 30)
        self.assertEqual(motif.alignment.sequences[10].length, 30)
        self.assertEqual(motif.alignment.sequences[11].length, 30)
        self.assertEqual(motif.alignment.sequences[12].length, 30)
        self.assertEqual(motif.alignment.sequences[13].length, 30)
        self.assertEqual(motif.alignment.sequences[14].length, 30)
        self.assertEqual(motif.alignment.sequences[15].length, 30)
        self.assertEqual(motif.alignment.sequences[16].length, 30)
        self.assertEqual(motif.alignment.sequences[17].length, 30)
        self.assertEqual(motif.alignment.sequences[18].length, 30)
        self.assertEqual(motif.alignment.sequences[19].length, 30)
        self.assertEqual(motif.alignment.sequences[20].length, 30)
        self.assertEqual(motif.alignment.sequences[21].length, 30)
        self.assertEqual(motif.alignment.sequences[22].length, 30)
        self.assertEqual(motif.alignment.sequences[23].length, 30)
        self.assertEqual(motif.alignment.sequences[0].start, 223)
        self.assertEqual(motif.alignment.sequences[1].start, 227)
        self.assertEqual(motif.alignment.sequences[2].start, 275)
        self.assertEqual(motif.alignment.sequences[3].start, 237)
        self.assertEqual(motif.alignment.sequences[4].start, 138)
        self.assertEqual(motif.alignment.sequences[5].start, 179)
        self.assertEqual(motif.alignment.sequences[6].start, 131)
        self.assertEqual(motif.alignment.sequences[7].start, 172)
        self.assertEqual(motif.alignment.sequences[8].start, 276)
        self.assertEqual(motif.alignment.sequences[9].start, 124)
        self.assertEqual(motif.alignment.sequences[10].start, 247)
        self.assertEqual(motif.alignment.sequences[11].start, 272)
        self.assertEqual(motif.alignment.sequences[12].start, 145)
        self.assertEqual(motif.alignment.sequences[13].start, 286)
        self.assertEqual(motif.alignment.sequences[14].start, 296)
        self.assertEqual(motif.alignment.sequences[15].start, 348)
        self.assertEqual(motif.alignment.sequences[16].start, 83)
        self.assertEqual(motif.alignment.sequences[17].start, 189)
        self.assertEqual(motif.alignment.sequences[18].start, 73)
        self.assertEqual(motif.alignment.sequences[19].start, 205)
        self.assertEqual(motif.alignment.sequences[20].start, 198)
        self.assertEqual(motif.alignment.sequences[21].start, 327)
        self.assertEqual(motif.alignment.sequences[22].start, 349)
        self.assertEqual(motif.alignment.sequences[23].start, 24)
        self.assertEqual(motif.alignment.sequences[0], "GGLNGRPSKLPDVCYSWWVLSSLAIIGRLD")
        self.assertEqual(motif.alignment.sequences[1], "GGLNGRPEKLPDVCYSWWVLASLKIIGRLH")
        self.assertEqual(motif.alignment.sequences[2], "GGFQGRENKFADTCYAFWCLNSLHLLTKDW")
        self.assertEqual(motif.alignment.sequences[3], "GGIGGVPGMEAHGGYTFCGLAALVILKKER")
        self.assertEqual(motif.alignment.sequences[4], "GGFGGGPGQYPHLAPTYAAVNALCIIGTEE")
        self.assertEqual(motif.alignment.sequences[5], "GGFGCRPGSESHAGQIYCCTGFLAITSQLH")
        self.assertEqual(motif.alignment.sequences[6], "GSFAGDIWGEIDTRFSFCAVATLALLGKLD")
        self.assertEqual(motif.alignment.sequences[7], "GGFGLCPNAESHAAQAFTCLGALAIANKLD")
        self.assertEqual(motif.alignment.sequences[8], "GGFADRPGDMVDPFHTLFGIAGLSLLGEEQ")
        self.assertEqual(motif.alignment.sequences[9], "GSFQGDRFGEVDTRFVYTALSALSILGELT")
        self.assertEqual(
            motif.alignment.sequences[10], "GFGSCPHVDEAHGGYTFCATASLAILRSMD"
        )
        self.assertEqual(
            motif.alignment.sequences[11], "GGISDRPENEVDVFHTVFGVAGLSLMGYDN"
        )
        self.assertEqual(
            motif.alignment.sequences[12], "GPFGGGPGQLSHLASTYAAINALSLCDNID"
        )
        self.assertEqual(
            motif.alignment.sequences[13], "GGFQGRCNKLVDGCYSFWQAGLLPLLHRAL"
        )
        self.assertEqual(
            motif.alignment.sequences[14], "RGFCGRSNKLVDGCYSFWVGGSAAILEAFG"
        )
        self.assertEqual(
            motif.alignment.sequences[15], "GGLLDKPGKSRDFYHTCYCLSGLSIAQHFG"
        )
        self.assertEqual(
            motif.alignment.sequences[16], "GGVSASIGHDPHLLYTLSAVQILTLYDSIH"
        )
        self.assertEqual(
            motif.alignment.sequences[17], "GSFLMHVGGEVDVRSAYCAASVASLTNIIT"
        )
        self.assertEqual(
            motif.alignment.sequences[18], "GAFAPFPRHDAHLLTTLSAVQILATYDALD"
        )
        self.assertEqual(
            motif.alignment.sequences[19], "YNGAFGAHNEPHSGYTSCALSTLALLSSLE"
        )
        self.assertEqual(
            motif.alignment.sequences[20], "GFKTCLEVGEVDTRGIYCALSIATLLNILT"
        )
        self.assertEqual(
            motif.alignment.sequences[21], "GGFSKNDEEDADLYHSCLGSAALALIEGKF"
        )
        self.assertEqual(
            motif.alignment.sequences[22], "PGLRDKPGAHSDFYHTNYCLLGLAVAESSY"
        )
        self.assertEqual(
            motif.alignment.sequences[23], "HNFEYWLTEHLRLNGIYWGLTALCVLDSPE"
        )
        self.assertEqual(motif.consensus, "GGFGGRPGKEVDLCYTFCALAALALLGSLD")
        self.assertEqual(motif[3:-8].consensus, "GGRPGKEVDLCYTFCALAA")
        motif = record[1]
        self.assertEqual(motif.name, "JNKEKLLEYILSCQ")
        self.assertEqual(record["JNKEKLLEYILSCQ"], motif)
        self.assertEqual(motif.num_occurrences, 21)
        self.assertAlmostEqual(motif.evalue, 6.1e-21, places=22)
        self.assertEqual(motif.alphabet, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(motif.alignment.sequences), 21)
        self.assertAlmostEqual(motif.alignment.sequences[0].pvalue, 2.71e-12, places=14)
        self.assertAlmostEqual(motif.alignment.sequences[1].pvalue, 5.7e-12, places=13)
        self.assertAlmostEqual(motif.alignment.sequences[2].pvalue, 6.43e-12, places=14)
        self.assertAlmostEqual(motif.alignment.sequences[3].pvalue, 2.61e-11, places=13)
        self.assertAlmostEqual(motif.alignment.sequences[4].pvalue, 6.3e-11, places=12)
        self.assertAlmostEqual(motif.alignment.sequences[5].pvalue, 2.7e-10, places=11)
        self.assertAlmostEqual(motif.alignment.sequences[6].pvalue, 4.03e-10, places=12)
        self.assertAlmostEqual(motif.alignment.sequences[7].pvalue, 1.27e-09, places=11)
        self.assertAlmostEqual(motif.alignment.sequences[8].pvalue, 3.17e-09, places=11)
        self.assertAlmostEqual(motif.alignment.sequences[9].pvalue, 6.39e-09, places=11)
        self.assertAlmostEqual(
            motif.alignment.sequences[10].pvalue, 6.96e-09, places=11
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[11].pvalue, 1.06e-08, places=10
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[12].pvalue, 1.26e-08, places=10
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[13].pvalue, 1.37e-08, places=10
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[14].pvalue, 2.07e-08, places=10
        )
        self.assertAlmostEqual(
            motif.alignment.sequences[15].pvalue, 4.96e-08, places=10
        )
        self.assertAlmostEqual(motif.alignment.sequences[16].pvalue, 1.15e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[17].pvalue, 1.44e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[18].pvalue, 1.55e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[19].pvalue, 1.93e-07, places=9)
        self.assertAlmostEqual(motif.alignment.sequences[20].pvalue, 5.2e-07, places=8)
        self.assertEqual(motif.alignment.sequences[0].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[1].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[2].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[3].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[4].sequence_name, "CAL1_YEAST")
        self.assertEqual(motif.alignment.sequences[5].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[6].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[7].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[8].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[9].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[10].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[11].sequence_name, "CAL1_YEAST")
        self.assertEqual(motif.alignment.sequences[12].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[13].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[14].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[15].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[16].sequence_name, "RATRABGERB")
        self.assertEqual(motif.alignment.sequences[17].sequence_name, "RAM1_YEAST")
        self.assertEqual(motif.alignment.sequences[18].sequence_name, "PFTB_RAT")
        self.assertEqual(motif.alignment.sequences[19].sequence_name, "BET2_YEAST")
        self.assertEqual(motif.alignment.sequences[20].sequence_name, "CAL1_YEAST")
        self.assertEqual(motif.alignment.sequences[0].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[1].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[2].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[3].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[4].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[5].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[6].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[7].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[8].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[9].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[10].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[11].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[12].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[13].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[14].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[15].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[16].sequence_id, "sequence_3")
        self.assertEqual(motif.alignment.sequences[17].sequence_id, "sequence_0")
        self.assertEqual(motif.alignment.sequences[18].sequence_id, "sequence_1")
        self.assertEqual(motif.alignment.sequences[19].sequence_id, "sequence_2")
        self.assertEqual(motif.alignment.sequences[20].sequence_id, "sequence_4")
        self.assertEqual(motif.alignment.sequences[0].strand, "+")
        self.assertEqual(motif.alignment.sequences[1].strand, "+")
        self.assertEqual(motif.alignment.sequences[2].strand, "+")
        self.assertEqual(motif.alignment.sequences[3].strand, "+")
        self.assertEqual(motif.alignment.sequences[4].strand, "+")
        self.assertEqual(motif.alignment.sequences[5].strand, "+")
        self.assertEqual(motif.alignment.sequences[6].strand, "+")
        self.assertEqual(motif.alignment.sequences[7].strand, "+")
        self.assertEqual(motif.alignment.sequences[8].strand, "+")
        self.assertEqual(motif.alignment.sequences[9].strand, "+")
        self.assertEqual(motif.alignment.sequences[10].strand, "+")
        self.assertEqual(motif.alignment.sequences[11].strand, "+")
        self.assertEqual(motif.alignment.sequences[12].strand, "+")
        self.assertEqual(motif.alignment.sequences[13].strand, "+")
        self.assertEqual(motif.alignment.sequences[14].strand, "+")
        self.assertEqual(motif.alignment.sequences[15].strand, "+")
        self.assertEqual(motif.alignment.sequences[16].strand, "+")
        self.assertEqual(motif.alignment.sequences[17].strand, "+")
        self.assertEqual(motif.alignment.sequences[18].strand, "+")
        self.assertEqual(motif.alignment.sequences[19].strand, "+")
        self.assertEqual(motif.alignment.sequences[20].strand, "+")
        self.assertEqual(motif.alignment.sequences[0].length, 14)
        self.assertEqual(motif.alignment.sequences[1].length, 14)
        self.assertEqual(motif.alignment.sequences[2].length, 14)
        self.assertEqual(motif.alignment.sequences[3].length, 14)
        self.assertEqual(motif.alignment.sequences[4].length, 14)
        self.assertEqual(motif.alignment.sequences[5].length, 14)
        self.assertEqual(motif.alignment.sequences[6].length, 14)
        self.assertEqual(motif.alignment.sequences[7].length, 14)
        self.assertEqual(motif.alignment.sequences[8].length, 14)
        self.assertEqual(motif.alignment.sequences[9].length, 14)
        self.assertEqual(motif.alignment.sequences[10].length, 14)
        self.assertEqual(motif.alignment.sequences[11].length, 14)
        self.assertEqual(motif.alignment.sequences[12].length, 14)
        self.assertEqual(motif.alignment.sequences[13].length, 14)
        self.assertEqual(motif.alignment.sequences[14].length, 14)
        self.assertEqual(motif.alignment.sequences[15].length, 14)
        self.assertEqual(motif.alignment.sequences[16].length, 14)
        self.assertEqual(motif.alignment.sequences[17].length, 14)
        self.assertEqual(motif.alignment.sequences[18].length, 14)
        self.assertEqual(motif.alignment.sequences[19].length, 14)
        self.assertEqual(motif.alignment.sequences[20].length, 14)
        self.assertEqual(motif.alignment.sequences[0].start, 66)
        self.assertEqual(motif.alignment.sequences[1].start, 254)
        self.assertEqual(motif.alignment.sequences[2].start, 258)
        self.assertEqual(motif.alignment.sequences[3].start, 162)
        self.assertEqual(motif.alignment.sequences[4].start, 190)
        self.assertEqual(motif.alignment.sequences[5].start, 278)
        self.assertEqual(motif.alignment.sequences[6].start, 172)
        self.assertEqual(motif.alignment.sequences[7].start, 114)
        self.assertEqual(motif.alignment.sequences[8].start, 7)
        self.assertEqual(motif.alignment.sequences[9].start, 268)
        self.assertEqual(motif.alignment.sequences[10].start, 414)
        self.assertEqual(motif.alignment.sequences[11].start, 126)
        self.assertEqual(motif.alignment.sequences[12].start, 220)
        self.assertEqual(motif.alignment.sequences[13].start, 55)
        self.assertEqual(motif.alignment.sequences[14].start, 229)
        self.assertEqual(motif.alignment.sequences[15].start, 330)
        self.assertEqual(motif.alignment.sequences[16].start, 18)
        self.assertEqual(motif.alignment.sequences[17].start, 180)
        self.assertEqual(motif.alignment.sequences[18].start, 73)
        self.assertEqual(motif.alignment.sequences[19].start, 107)
        self.assertEqual(motif.alignment.sequences[20].start, 36)
        self.assertEqual(motif.alignment.sequences[0], "MNKEEILVFIKSCQ")
        self.assertEqual(motif.alignment.sequences[1], "INYEKLTEFILKCQ")
        self.assertEqual(motif.alignment.sequences[2], "IDREKLRSFILACQ")
        self.assertEqual(motif.alignment.sequences[3], "INVEKAIEFVLSCM")
        self.assertEqual(motif.alignment.sequences[4], "IDTEKLLGYIMSQQ")
        self.assertEqual(motif.alignment.sequences[5], "INVEKLLEWSSARQ")
        self.assertEqual(motif.alignment.sequences[6], "INREKLLQYLYSLK")
        self.assertEqual(motif.alignment.sequences[7], "INVDKVVAYVQSLQ")
        self.assertEqual(motif.alignment.sequences[8], "LLKEKHIRYIESLD")
        self.assertEqual(motif.alignment.sequences[9], "LNLKSLLQWVTSRQ")
        self.assertEqual(motif.alignment.sequences[10], "ENVRKIIHYFKSNL")
        self.assertEqual(motif.alignment.sequences[11], "LDKRSLARFVSKCQ")
        self.assertEqual(motif.alignment.sequences[12], "DLFEGTAEWIARCQ")
        self.assertEqual(motif.alignment.sequences[13], "FVKEEVISFVLSCW")
        self.assertEqual(motif.alignment.sequences[14], "ELTEGVLNYLKNCQ")
        self.assertEqual(motif.alignment.sequences[15], "FNKHALRDYILYCC")
        self.assertEqual(motif.alignment.sequences[16], "LLLEKHADYIASYG")
        self.assertEqual(motif.alignment.sequences[17], "IDRKGIYQWLISLK")
        self.assertEqual(motif.alignment.sequences[18], "LQREKHFHYLKRGL")
        self.assertEqual(motif.alignment.sequences[19], "DRKVRLISFIRGNQ")
        self.assertEqual(motif.alignment.sequences[20], "VNRMAIIFYSISGL")
        self.assertEqual(motif.consensus, "INKEKLIEYILSCQ")
        self.assertEqual(motif[3:-8].consensus, "EKL")

    def test_minimal_meme_parser(self):
        """Parse motifs/minimal_test.meme file."""
        with open("motifs/minimal_test.meme") as stream:
            record = motifs.parse(stream, "minimal")
        self.assertEqual(record.version, "4")
        self.assertEqual(record.alphabet, "ACGT")
        self.assertEqual(len(record.sequences), 0)
        self.assertEqual(record.command, "")
        self.assertEqual(len(record), 3)
        motif = record[0]
        self.assertEqual(motif.name, "KRP")
        self.assertEqual(record["KRP"], motif)
        self.assertEqual(motif.num_occurrences, 17)
        self.assertEqual(motif.length, 19)
        self.assertAlmostEqual(motif.background["A"], 0.30269730269730266)
        self.assertAlmostEqual(motif.background["C"], 0.1828171828171828)
        self.assertAlmostEqual(motif.background["G"], 0.20879120879120877)
        self.assertAlmostEqual(motif.background["T"], 0.30569430569430567)
        self.assertAlmostEqual(motif.evalue, 4.1e-09, places=10)
        self.assertEqual(motif.alphabet, "ACGT")
        self.assertIsNone(motif.alignment)
        self.assertEqual(motif.consensus, "TGTGATCGAGGTCACACTT")
        self.assertEqual(motif.degenerate_consensus, "TGTGANNNWGNTCACAYWW")
        self.assertTrue(
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
        self.assertEqual(motif[2:9].consensus, "TGATCGA")
        motif = record[1]
        self.assertEqual(motif.name, "IFXA")
        self.assertEqual(record["IFXA"], motif)
        self.assertEqual(motif.num_occurrences, 14)
        self.assertEqual(motif.length, 18)
        self.assertAlmostEqual(motif.background["A"], 0.30269730269730266)
        self.assertAlmostEqual(motif.background["C"], 0.1828171828171828)
        self.assertAlmostEqual(motif.background["G"], 0.20879120879120877)
        self.assertAlmostEqual(motif.background["T"], 0.30569430569430567)
        self.assertAlmostEqual(motif.evalue, 3.2e-35, places=36)
        self.assertEqual(motif.alphabet, "ACGT")
        self.assertIsNone(motif.alignment)
        self.assertEqual(motif.consensus, "TACTGTATATATATCCAG")
        self.assertEqual(motif.degenerate_consensus, "TACTGTATATAHAWMCAG")
        self.assertTrue(
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
        self.assertEqual(motif[2:9].consensus, "CTGTATA")
        with open("motifs/minimal_test.meme") as stream:
            record = motifs.parse(stream, "minimal")
        motif = record[2]
        self.assertEqual(motif.name, "IFXA_no_nsites_no_evalue")
        self.assertEqual(record["IFXA_no_nsites_no_evalue"], motif)
        self.assertEqual(motif.num_occurrences, 20)
        self.assertEqual(motif.length, 18)
        self.assertAlmostEqual(motif.background["A"], 0.30269730269730266)
        self.assertAlmostEqual(motif.background["C"], 0.1828171828171828)
        self.assertAlmostEqual(motif.background["G"], 0.20879120879120877)
        self.assertAlmostEqual(motif.background["T"], 0.30569430569430567)
        self.assertAlmostEqual(motif.evalue, 0.0, places=36)
        self.assertEqual(motif.alphabet, "ACGT")
        self.assertIsNone(motif.alignment)
        self.assertEqual(motif.consensus, "TACTGTATATATATCCAG")
        self.assertEqual(motif.degenerate_consensus, "TACTGTATATAHAWMCAG")
        self.assertTrue(
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
        self.assertEqual(motif[2:9].consensus, "CTGTATA")

    def test_meme_parser_rna(self):
        """Test if Bio.motifs can parse MEME output files using RNA."""
        with open("motifs/minimal_test_rna.meme") as stream:
            record = motifs.parse(stream, "minimal")
        self.assertEqual(record.version, "4")
        self.assertEqual(record.alphabet, "ACGU")
        self.assertEqual(len(record.sequences), 0)
        self.assertEqual(record.command, "")
        self.assertEqual(len(record), 3)
        motif = record[0]
        self.assertEqual(motif.name, "KRP_fake_RNA")
        self.assertEqual(record["KRP_fake_RNA"], motif)
        self.assertEqual(motif.num_occurrences, 17)
        self.assertEqual(motif.length, 19)
        self.assertAlmostEqual(motif.background["A"], 0.30269730269730266)
        self.assertAlmostEqual(motif.background["C"], 0.1828171828171828)
        self.assertAlmostEqual(motif.background["G"], 0.20879120879120877)
        self.assertAlmostEqual(motif.background["U"], 0.30569430569430567)
        self.assertAlmostEqual(motif.evalue, 4.1e-09, places=10)
        self.assertEqual(motif.alphabet, "ACGU")
        self.assertIsNone(motif.alignment)
        self.assertEqual(motif.consensus, "UGUGAUCGAGGUCACACUU")
        self.assertEqual(motif.degenerate_consensus, "UGUGANNNWGNUCACAYWW")
        self.assertTrue(
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
        self.assertEqual(motif[2:9].consensus, "UGAUCGA")
        motif = record[1]
        self.assertEqual(motif.name, "IFXA_fake_RNA")
        self.assertEqual(record["IFXA_fake_RNA"], motif)
        self.assertEqual(motif.num_occurrences, 14)
        self.assertEqual(motif.length, 18)
        self.assertAlmostEqual(motif.background["A"], 0.30269730269730266)
        self.assertAlmostEqual(motif.background["C"], 0.1828171828171828)
        self.assertAlmostEqual(motif.background["G"], 0.20879120879120877)
        self.assertAlmostEqual(motif.background["U"], 0.30569430569430567)
        self.assertAlmostEqual(motif.evalue, 3.2e-35, places=36)
        self.assertEqual(motif.alphabet, "ACGU")
        self.assertIsNone(motif.alignment)
        self.assertEqual(motif.consensus, "UACUGUAUAUAUAUCCAG")
        self.assertEqual(motif.degenerate_consensus, "UACUGUAUAUAHAWMCAG")
        self.assertTrue(
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
        self.assertEqual(motif[2:9].consensus, "CUGUAUA")

        motif = record[2]
        self.assertEqual(motif.name, "IFXA_no_nsites_no_evalue_fake_RNA")
        self.assertEqual(record["IFXA_no_nsites_no_evalue_fake_RNA"], motif)
        self.assertEqual(motif.num_occurrences, 20)
        self.assertEqual(motif.length, 18)
        self.assertAlmostEqual(motif.background["A"], 0.30269730269730266)
        self.assertAlmostEqual(motif.background["C"], 0.1828171828171828)
        self.assertAlmostEqual(motif.background["G"], 0.20879120879120877)
        self.assertAlmostEqual(motif.background["U"], 0.30569430569430567)
        self.assertAlmostEqual(motif.evalue, 0.0, places=36)
        self.assertEqual(motif.alphabet, "ACGU")
        self.assertIsNone(motif.alignment)
        self.assertEqual(motif.consensus, "UACUGUAUAUAUAUCCAG")
        self.assertEqual(motif.degenerate_consensus, "UACUGUAUAUAHAWMCAG")
        self.assertTrue(
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
        self.assertEqual(motif[2:9].consensus, "CUGUAUA")


class TestMAST(unittest.TestCase):

    """Transfac format tests."""

    def test_transfac_parser(self):
        """Parse motifs/transfac.dat file."""
        with open("motifs/transfac.dat") as stream:
            record = motifs.parse(stream, "TRANSFAC")
        motif = record[0]
        self.assertEqual(motif["ID"], "motif1")
        self.assertEqual(len(motif.counts), 4)
        self.assertEqual(motif.counts.length, 12)
        self.assertEqual(motif.counts["A", 0], 1)
        self.assertEqual(motif.counts["A", 1], 2)
        self.assertEqual(motif.counts["A", 2], 3)
        self.assertEqual(motif.counts["A", 3], 0)
        self.assertEqual(motif.counts["A", 4], 5)
        self.assertEqual(motif.counts["A", 5], 0)
        self.assertEqual(motif.counts["A", 6], 0)
        self.assertEqual(motif.counts["A", 7], 0)
        self.assertEqual(motif.counts["A", 8], 0)
        self.assertEqual(motif.counts["A", 9], 0)
        self.assertEqual(motif.counts["A", 10], 0)
        self.assertEqual(motif.counts["A", 11], 1)
        self.assertEqual(motif.counts["C", 0], 2)
        self.assertEqual(motif.counts["C", 1], 1)
        self.assertEqual(motif.counts["C", 2], 0)
        self.assertEqual(motif.counts["C", 3], 5)
        self.assertEqual(motif.counts["C", 4], 0)
        self.assertEqual(motif.counts["C", 5], 0)
        self.assertEqual(motif.counts["C", 6], 1)
        self.assertEqual(motif.counts["C", 7], 0)
        self.assertEqual(motif.counts["C", 8], 0)
        self.assertEqual(motif.counts["C", 9], 1)
        self.assertEqual(motif.counts["C", 10], 2)
        self.assertEqual(motif.counts["C", 11], 0)
        self.assertEqual(motif.counts["G", 0], 2)
        self.assertEqual(motif.counts["G", 1], 2)
        self.assertEqual(motif.counts["G", 2], 1)
        self.assertEqual(motif.counts["G", 3], 0)
        self.assertEqual(motif.counts["G", 4], 0)
        self.assertEqual(motif.counts["G", 5], 4)
        self.assertEqual(motif.counts["G", 6], 4)
        self.assertEqual(motif.counts["G", 7], 0)
        self.assertEqual(motif.counts["G", 8], 5)
        self.assertEqual(motif.counts["G", 9], 2)
        self.assertEqual(motif.counts["G", 10], 0)
        self.assertEqual(motif.counts["G", 11], 3)
        self.assertEqual(motif.counts["T", 0], 0)
        self.assertEqual(motif.counts["T", 1], 0)
        self.assertEqual(motif.counts["T", 2], 1)
        self.assertEqual(motif.counts["T", 3], 0)
        self.assertEqual(motif.counts["T", 4], 0)
        self.assertEqual(motif.counts["T", 5], 1)
        self.assertEqual(motif.counts["T", 6], 0)
        self.assertEqual(motif.counts["T", 7], 5)
        self.assertEqual(motif.counts["T", 8], 0)
        self.assertEqual(motif.counts["T", 9], 2)
        self.assertEqual(motif.counts["T", 10], 3)
        self.assertEqual(motif.counts["T", 11], 1)
        self.assertEqual(motif.degenerate_consensus, "SRACAGGTGKYG")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.4780719051126377,
                        0.4780719051126377,
                        0.6290494055453314,
                        2.0,
                        2.0,
                        1.278071905112638,
                        1.278071905112638,
                        2.0,
                        2.0,
                        0.4780719051126377,
                        1.0290494055453312,
                        0.6290494055453314,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-2].degenerate_consensus, "RACAGGTGK")
        self.assertTrue(
            np.allclose(
                motif[1:-2].relative_entropy,
                np.array(
                    [
                        0.4780719051126377,
                        0.6290494055453314,
                        2.0,
                        2.0,
                        1.278071905112638,
                        1.278071905112638,
                        2.0,
                        2.0,
                        0.4780719051126377,
                    ]
                ),
            )
        )
        motif = record[1]
        self.assertEqual(motif["ID"], "motif2")
        self.assertEqual(len(motif.counts), 4)
        self.assertEqual(motif.counts.length, 10)
        self.assertEqual(motif.counts["A", 0], 2)
        self.assertEqual(motif.counts["A", 1], 1)
        self.assertEqual(motif.counts["A", 2], 0)
        self.assertEqual(motif.counts["A", 3], 3)
        self.assertEqual(motif.counts["A", 4], 0)
        self.assertEqual(motif.counts["A", 5], 5)
        self.assertEqual(motif.counts["A", 6], 0)
        self.assertEqual(motif.counts["A", 7], 0)
        self.assertEqual(motif.counts["A", 8], 0)
        self.assertEqual(motif.counts["A", 9], 0)
        self.assertEqual(motif.counts["C", 0], 1)
        self.assertEqual(motif.counts["C", 1], 2)
        self.assertEqual(motif.counts["C", 2], 5)
        self.assertEqual(motif.counts["C", 3], 0)
        self.assertEqual(motif.counts["C", 4], 0)
        self.assertEqual(motif.counts["C", 5], 0)
        self.assertEqual(motif.counts["C", 6], 1)
        self.assertEqual(motif.counts["C", 7], 0)
        self.assertEqual(motif.counts["C", 8], 0)
        self.assertEqual(motif.counts["C", 9], 2)
        self.assertEqual(motif.counts["G", 0], 2)
        self.assertEqual(motif.counts["G", 1], 2)
        self.assertEqual(motif.counts["G", 2], 0)
        self.assertEqual(motif.counts["G", 3], 1)
        self.assertEqual(motif.counts["G", 4], 4)
        self.assertEqual(motif.counts["G", 5], 0)
        self.assertEqual(motif.counts["G", 6], 4)
        self.assertEqual(motif.counts["G", 7], 5)
        self.assertEqual(motif.counts["G", 8], 0)
        self.assertEqual(motif.counts["G", 9], 0)
        self.assertEqual(motif.counts["T", 0], 0)
        self.assertEqual(motif.counts["T", 1], 0)
        self.assertEqual(motif.counts["T", 2], 0)
        self.assertEqual(motif.counts["T", 3], 1)
        self.assertEqual(motif.counts["T", 4], 1)
        self.assertEqual(motif.counts["T", 5], 0)
        self.assertEqual(motif.counts["T", 6], 0)
        self.assertEqual(motif.counts["T", 7], 0)
        self.assertEqual(motif.counts["T", 8], 5)
        self.assertEqual(motif.counts["T", 9], 3)
        self.assertEqual(motif.degenerate_consensus, "RSCAGAGGTY")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.4780719051126377,
                        0.4780719051126377,
                        2.0,
                        0.6290494055453314,
                        1.278071905112638,
                        2.0,
                        1.278071905112638,
                        2.0,
                        2.0,
                        1.0290494055453312,
                    ]
                ),
            )
        )
        self.assertEqual(motif[::2].degenerate_consensus, "RCGGT")
        self.assertTrue(
            np.allclose(
                motif[::2].relative_entropy,
                np.array(
                    [0.4780719051126377, 2.0, 1.278071905112638, 1.278071905112638, 2.0]
                ),
            )
        )

    def test_permissive_transfac_parser(self):
        """Parse the TRANSFAC-like file motifs/MA0056.1.transfac."""
        # The test file MA0056.1.transfac was obtained from the JASPAR database
        # in a TRANSFAC-like format.
        # Khan, A. et al. JASPAR 2018: update of the open-access database of
        # transcription factor binding profiles and its web framework.
        # Nucleic Acids Res. 2018; 46:D260-D266,
        path = "motifs/MA0056.1.transfac"
        with open(path) as stream:
            self.assertRaises(ValueError, motifs.parse, stream, "TRANSFAC")
        with open(path) as stream:
            records = motifs.parse(stream, "TRANSFAC", strict=False)
        motif = records[0]
        self.assertEqual(sorted(motif.keys()), ["AC", "DE", "ID"])
        self.assertEqual(motif["AC"], "MA0056.1")
        self.assertEqual(motif["DE"], "MA0056.1 MZF1 ; From JASPAR 2018")
        self.assertEqual(motif["ID"], "MZF1")
        self.assertEqual(motif.counts.length, 6)
        self.assertEqual(len(motif.counts), 4)
        self.assertEqual(motif.counts["A", 0], 3.0)
        self.assertEqual(motif.counts["A", 1], 0.0)
        self.assertEqual(motif.counts["A", 2], 2.0)
        self.assertEqual(motif.counts["A", 3], 0.0)
        self.assertEqual(motif.counts["A", 4], 0.0)
        self.assertEqual(motif.counts["A", 5], 18.0)
        self.assertEqual(motif.counts["C", 0], 5.0)
        self.assertEqual(motif.counts["C", 1], 0.0)
        self.assertEqual(motif.counts["C", 2], 0.0)
        self.assertEqual(motif.counts["C", 3], 0.0)
        self.assertEqual(motif.counts["C", 4], 0.0)
        self.assertEqual(motif.counts["C", 5], 0.0)
        self.assertEqual(motif.counts["G", 0], 4.0)
        self.assertEqual(motif.counts["G", 1], 19.0)
        self.assertEqual(motif.counts["G", 2], 18.0)
        self.assertEqual(motif.counts["G", 3], 19.0)
        self.assertEqual(motif.counts["G", 4], 20.0)
        self.assertEqual(motif.counts["G", 5], 2.0)
        self.assertEqual(motif.counts["T", 0], 8.0)
        self.assertEqual(motif.counts["T", 1], 1.0)
        self.assertEqual(motif.counts["T", 2], 0.0)
        self.assertEqual(motif.counts["T", 3], 1.0)
        self.assertEqual(motif.counts["T", 4], 0.0)
        self.assertEqual(motif.counts["T", 5], 0.0)
        self.assertEqual(motif.consensus, "TGGGGA")
        self.assertEqual(motif.degenerate_consensus, "NGGGGA")
        self.assertTrue(
            np.allclose(
                motif.relative_entropy,
                np.array(
                    [
                        0.09629830394265171,
                        1.7136030428840439,
                        1.5310044064107189,
                        1.7136030428840439,
                        2.0,
                        1.5310044064107189,
                    ]
                ),
            )
        )
        self.assertEqual(motif[1:-3].degenerate_consensus, "GG")
        self.assertTrue(
            np.allclose(
                motif[1:-3].relative_entropy,
                np.array([1.7136030428840439, 1.5310044064107189]),
            )
        )

    def test_TFoutput(self):
        """Ensure that we can write proper TransFac output files."""
        m = motifs.create([Seq("ATATA")])
        with tempfile.TemporaryFile("w") as stream:
            stream.write(format(m, "transfac"))


class MotifTestPWM(unittest.TestCase):