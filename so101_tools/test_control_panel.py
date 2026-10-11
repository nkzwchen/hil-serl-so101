"""No-hardware input routing and label GUI action tests."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from control_panel import KeyGate
from label_demos import Labeler


class PanelTests(unittest.TestCase):
    def test_old_phase_and_stale_keys_ignored(self):
        g=KeyGate();g.set_phase('READY')
        old=g.generation;g.set_phase('RETURN')
        for gen,stamp in [(old,time.monotonic()),(g.generation,time.monotonic()-2)]:
            g.receive(('key',gen,stamp,'c'))
        self.assertIsNone(g.read())
        g.receive(('key',g.generation,time.monotonic(),'c'))
        self.assertEqual(g.read(),'c')
        self.assertIsNone(g.read())

    def test_upload_keys_disabled_close_persists(self):
        g=KeyGate();g.set_phase('REVIEW')
        g.receive(('key',g.generation,time.monotonic(),' '))
        self.assertIsNone(g.read())
        g.receive(('close',));g.set_phase('PRECISE')
        self.assertEqual(g.read(),'x')

    def test_unreviewed_confirm_is_inline_readable_text(self):
        app=Labeler.__new__(Labeler)
        app.paths=['unused'];app.fi=0;app.review_once=True;app.labels={}
        app.lbl_info=Mock()
        self.assertIsNone(app._save())
        text=app.lbl_info.configure.call_args.kwargs['text']
        self.assertIn('请先',text)
        self.assertNotIn('\\u',text)

    def test_skip_does_not_confirm(self):
        app=Labeler.__new__(Labeler)
        app.review_once=True;app.labels={};app.review_confirmed=False
        app.root=SimpleNamespace(destroy=Mock())
        app._skip_upload()
        self.assertTrue(app.review_skipped)
        self.assertFalse(app.review_confirmed)
        app.root.destroy.assert_called_once()

if __name__=='__main__':unittest.main()
