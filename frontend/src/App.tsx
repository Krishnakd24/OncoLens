import { useEffect, useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Rnd } from 'react-rnd';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [config, setConfig] = useState<any>(null);
  const [probeX, setProbeX] = useState('');
  const [probeY, setProbeY] = useState('');
  const [displayMode, setDisplayMode] = useState('both');
  
  const [contourData, setContourData] = useState<any>(null);
  const [hotspotsData, setHotspotsData] = useState<any>(null);
  const [networkData, setNetworkData] = useState<any>(null);
  const [profilesData, setProfilesData] = useState<any>(null);

  useEffect(() => {
    axios.get(`${API_BASE}/api/config`).then(res => {
      setConfig(res.data);
      setProbeX(res.data.default_x);
      setProbeY(res.data.default_y);
    }).catch(err => console.error(err));
  }, []);

  useEffect(() => {
    if (!probeX || !probeY) return;
    
    // Fetch Contour
    axios.post(`${API_BASE}/api/contour`, { probe_x: probeX, probe_y: probeY, display_mode: displayMode })
      .then(res => {
        res.data.figure.layout.autosize = true;
        res.data.figure.layout.margin = { l: 40, r: 40, t: 40, b: 40 };
        if (res.data.figure.layout.height) delete res.data.figure.layout.height;
        if (res.data.figure.layout.width) delete res.data.figure.layout.width;
        setContourData(res.data);
      });
      
    // Fetch Hotspots
    axios.post(`${API_BASE}/api/hotspots`, { selected_chr: "All" })
      .then(res => {
        res.data.figure.layout.autosize = true;
        if (res.data.figure.layout.height) delete res.data.figure.layout.height;
        if (res.data.figure.layout.width) delete res.data.figure.layout.width;
        setHotspotsData(res.data);
      });
      
    // Fetch Network
    axios.post(`${API_BASE}/api/network`, { selected_probe: probeX })
      .then(res => {
        res.data.figure.layout.autosize = true;
        if (res.data.figure.layout.height) delete res.data.figure.layout.height;
        if (res.data.figure.layout.width) delete res.data.figure.layout.width;
        setNetworkData(res.data);
      });
      
    // Fetch Profiles
    axios.post(`${API_BASE}/api/profiles`, { selected_probe: probeX })
      .then(res => {
        res.data.figure.layout.autosize = true;
        if (res.data.figure.layout.height) delete res.data.figure.layout.height;
        if (res.data.figure.layout.width) delete res.data.figure.layout.width;
        setProfilesData(res.data);
      });

  }, [probeX, probeY, displayMode]);

  if (!config) return <div>Loading WYSIWYG Workspace...</div>;

  const RndPanel = ({ children, defaultPos, title }: any) => (
    <Rnd
      default={defaultPos}
      minWidth={300}
      minHeight={200}
      bounds="parent"
      dragHandleClassName="panel-header"
      style={{ 
        border: '1px solid #cbd5e1', 
        backgroundColor: 'white', 
        boxShadow: '0 4px 15px rgba(0,0,0,0.05)',
        borderRadius: 8,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <div className="panel-header" style={{ padding: '8px 15px', backgroundColor: '#f1f5f9', cursor: 'move', fontWeight: 'bold', fontSize: '0.9rem', borderBottom: '1px solid #cbd5e1' }}>
        {title} (Drag Here)
      </div>
      <div style={{ flex: 1, position: 'relative', width: '100%', height: '100%' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
          {children}
        </div>
      </div>
    </Rnd>
  );

  return (
    <div style={{ fontFamily: 'Outfit, sans-serif', height: '100vh', width: '100vw', overflow: 'hidden', backgroundColor: '#f8fafc', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '15px 20px', backgroundColor: 'white', borderBottom: '1px solid #e2e8f0', zIndex: 10 }}>
        <h2 style={{ margin: '0 0 10px 0', fontSize: '1.2rem' }}>OncoLens WYSIWYG Canvas</h2>
        <div style={{ display: 'flex', gap: 15, fontSize: '0.9rem' }}>
          <div>
            <label style={{ fontWeight: 'bold', marginRight: 5 }}>Gene X: </label>
            <select value={probeX} onChange={e => setProbeX(e.target.value)} style={{ padding: 4, borderRadius: 4 }}>
              {config.gene_options.map((opt: any) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={{ fontWeight: 'bold', marginRight: 5 }}>Gene Y: </label>
            <select value={probeY} onChange={e => setProbeY(e.target.value)} style={{ padding: 4, borderRadius: 4 }}>
              {config.gene_options.map((opt: any) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={{ fontWeight: 'bold', marginRight: 5 }}>Display: </label>
            <select value={displayMode} onChange={e => setDisplayMode(e.target.value)} style={{ padding: 4, borderRadius: 4 }}>
              <option value="scatter">Scatter</option>
              <option value="contour">Contour</option>
              <option value="both">Both</option>
            </select>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        {contourData && (
          <RndPanel title="Contour/Scatter Plot" defaultPos={{ x: 20, y: 20, width: 700, height: 500 }}>
             <Plot data={contourData.figure.data} layout={contourData.figure.layout} useResizeHandler={true} style={{ width: '100%', height: '100%' }} />
          </RndPanel>
        )}
        
        {hotspotsData && (
          <RndPanel title="Chromosomal Hotspots" defaultPos={{ x: 740, y: 20, width: 500, height: 500 }}>
             <Plot data={hotspotsData.figure.data} layout={hotspotsData.figure.layout} useResizeHandler={true} style={{ width: '100%', height: '100%' }} />
          </RndPanel>
        )}
        
        {networkData && (
          <RndPanel title="Co-expression Network" defaultPos={{ x: 20, y: 540, width: 500, height: 350 }}>
             <Plot data={networkData.figure.data} layout={networkData.figure.layout} useResizeHandler={true} style={{ width: '100%', height: '100%' }} />
          </RndPanel>
        )}
        
        {profilesData && (
          <RndPanel title="Expression Profiles" defaultPos={{ x: 540, y: 540, width: 700, height: 350 }}>
             <Plot data={profilesData.figure.data} layout={profilesData.figure.layout} useResizeHandler={true} style={{ width: '100%', height: '100%' }} />
          </RndPanel>
        )}
      </div>
    </div>
  );
}

export default App;

