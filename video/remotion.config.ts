import {existsSync} from 'node:fs';
import {Config} from '@remotion/cli/config';

// The bundled Headless Shell timed out on the measured Mac. A separate,
// headless Chrome process with a temporary profile rendered successfully.
// This does not connect to or use the user's logged-in browser profile.
const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = process.env.REMOTION_BROWSER_EXECUTABLE ||
  (process.platform === 'darwin' && existsSync(macChrome) ? macChrome : undefined);
if (browser) {
  Config.setBrowserExecutable(browser);
  Config.setChromeMode('chrome-for-testing');
}
Config.setChromiumOpenGlRenderer('swangle');
// Lossless intermediate frames keep chart text crisp.
Config.setVideoImageFormat('png');
