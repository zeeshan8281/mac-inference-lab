import React from 'react';
import {Composition, registerRoot} from 'remotion';
import {Dashboard} from './Dashboard';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/inter/800.css';

const Root: React.FC = () => (
  <Composition id="MacInferenceDashboard" component={Dashboard}
    durationInFrames={480} fps={30} width={1440} height={1080} />
);
registerRoot(Root);
