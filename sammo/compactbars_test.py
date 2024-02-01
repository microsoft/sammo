# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import io

import pytest

from sammo.compactbars import LinePrinter, SubProgressBar, CompactProgressBars


class TestLinePrinter:
    def setup_method(self, method):
        self.out = io.StringIO()
        self.lp = LinePrinter(out=self.out)

    def test_get_terminal_width(self):
        width = self.lp.get_terminal_width()
        assert width > 0

    def test_print(self):
        self.lp.print("Hello")
        assert self.out.getvalue() == "Hello"
        self.lp.print("World")
        assert self.out.getvalue() == "Hello\b\b\b\b\bWorld"

    def test_finalize(self):
        self.lp.finalize()
        assert self.out.getvalue() == "\n"


class TestSubProgressBar:
    def setup_method(self, method):
        self.parent = CompactProgressBars()
        self.spb = SubProgressBar(100, parent=self.parent)

    def test_invalid_total(self):
        with pytest.raises(ValueError):
            SubProgressBar(0, parent=self.parent)

    def test_default_values(self):
        assert self.spb._n_done == 0
        assert self.spb._total == 100

    def test_update(self):
        self.spb.update()
        assert self.spb._n_done == 1

    def test_str(self):
        result = str(self.spb)
        assert "[" in result
        assert "]" in result


class TestCompactProgressBars:
    def setup_method(self, method):
        self.cb = CompactProgressBars()

    def test_get_new_bar(self):
        bar = self.cb.get("test", 100)
        assert "test" in self.cb._bars

    def test_get_existing_bar(self):
        self.cb.get("test", 100)
        bar = self.cb.get("test", 200)
        assert bar.total == 200

    def test_str_representation(self):
        self.cb.get("test1", 100)
        self.cb.get("test2", 200)
        result = str(self.cb)
        assert ">>" in result

    def test_init_default_width(self):
        assert self.cb._width is not None

    def test_should_refresh(self):
        result = self.cb._should_refresh()
        assert result
