import unittest

from app.rendering import configure_renderer


class RenderingTests(unittest.TestCase):
    def test_observed_wayland_runtime_uses_software_without_disabling_sandbox(self):
        env = {"XDG_SESSION_TYPE": "wayland"}
        result = configure_renderer(webengine_version="6.9.2", environ=env, platform="linux")
        self.assertEqual(result["mode"], "software")
        self.assertEqual(env["QTWEBENGINE_CHROMIUM_FLAGS"], "--disable-gpu")
        self.assertNotIn("QTWEBENGINE_DISABLE_SANDBOX", env)

    def test_explicit_qpa_selection_overrides_session_for_auto_guard(self):
        for qpa in ("xcb", "offscreen", "minimal"):
            with self.subTest(qpa=qpa):
                env = {"QT_QPA_PLATFORM": qpa, "XDG_SESSION_TYPE": "wayland"}
                before = env.copy()
                self.assertEqual(configure_renderer(webengine_version="6.9.2", environ=env,
                                                    platform="linux")["mode"], "native")
                self.assertEqual(env, before)
        for qpa in ("wayland", "wayland-egl", "wayland:foo", "wayland;xcb"):
            with self.subTest(qpa=qpa):
                env = {"QT_QPA_PLATFORM": qpa, "XDG_SESSION_TYPE": "x11"}
                self.assertEqual(configure_renderer(webengine_version="6.9.2", environ=env,
                                                    platform="linux")["mode"], "software")

    def test_auto_policy_is_scoped_not_global(self):
        for system, version, session in (("win32", "6.9.2", "wayland"),
                                         ("linux", "6.9.3", "wayland"),
                                         ("linux", "6.9.20", "wayland"),
                                         ("linux", "6.9.2", "x11")):
            with self.subTest(system=system, version=version, session=session):
                env = {"XDG_SESSION_TYPE": session}
                self.assertEqual(configure_renderer(webengine_version=version, environ=env,
                                                    platform=system)["mode"], "native")
                self.assertNotIn("QTWEBENGINE_CHROMIUM_FLAGS", env)

    def test_flags_preserved_and_configuration_idempotent(self):
        env = {"QTWEBENGINE_CHROMIUM_FLAGS": '--enable-logging --user-agent="Test Browser"'}
        before = env["QTWEBENGINE_CHROMIUM_FLAGS"]
        configure_renderer("software", environ=env)
        configure_renderer("software", environ=env)
        self.assertEqual(env["QTWEBENGINE_CHROMIUM_FLAGS"], before + " --disable-gpu")

    def test_native_opt_out_preserves_existing_user_policy(self):
        env = {"XDG_SESSION_TYPE": "wayland"}
        self.assertEqual(configure_renderer("native", "6.9.2", env, "linux")["mode"], "native")
        self.assertNotIn("QTWEBENGINE_CHROMIUM_FLAGS", env)
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
        self.assertEqual(configure_renderer("native", "6.9.2", env, "linux")["mode"], "software")

    def test_software_explicitly_available_on_other_configs(self):
        env = {"XDG_SESSION_TYPE": "x11"}
        self.assertEqual(configure_renderer("software", "6.10.0", env, "linux")["mode"], "software")

    def test_wayland_display_is_an_alternative_detection(self):
        env = {"WAYLAND_DISPLAY": "wayland-0"}
        self.assertEqual(configure_renderer("auto", "6.9.2", env, "linux")["mode"], "software")

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            configure_renderer("no-sandbox", environ={})
