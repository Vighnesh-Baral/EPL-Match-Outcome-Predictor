import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, BarChart3,
  Check, ChevronDown, ChevronRight, CircleHelp, Crosshair, Database,
  ExternalLink, Gauge, LayoutDashboard, Menu, RefreshCw, Search, Shield,
  Sparkles, Target, Trophy, Users, X, CalendarDays
} from 'lucide-react';
import './styles.css';

const API = import.meta.env.VITE_API_BASE || '';
const HERO_IMAGE = 'https://images.pexels.com/photos/32285150/pexels-photo-32285150.jpeg?cs=srgb&dl=pexels-franco-monsalvo-252430633-32285150.jpg&fm=jpg';
const MATCH_IMAGE = 'https://images.pexels.com/photos/32179167/pexels-photo-32179167/free-photo-of-intense-football-match-with-players-in-action.jpeg?auto=compress&dpr=1&h=750&w=1260';

const number = (v, digits = 1) => v === null || v === undefined || v === '' || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(digits);
const safePct = v => Math.max(0, Math.min(100, Number(v) || 0));
const initials = name => String(name || '?').split(/\s+/).map(x => x[0]).join('').slice(0, 2).toUpperCase();
const toDate = value => value ? new Date(value).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) : 'Not available';
const formatNumber = value => value === null || value === undefined ? '—' : Number(value).toLocaleString('en-GB');

async function getJSON(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const body = await res.json().catch(() => ({ detail: 'Unexpected server response.' }));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

function useReveal(deps = []) {
  useEffect(() => {
    const nodes = document.querySelectorAll('.reveal:not(.is-visible)');
    if (!('IntersectionObserver' in window)) {
      nodes.forEach(node => node.classList.add('is-visible'));
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -32px 0px' });
    nodes.forEach(node => observer.observe(node));
    return () => observer.disconnect();
  }, deps);
}

function Brand({ onClick, inverse = false }) {
  return <button className={`brand ${inverse ? 'brand-inverse' : ''}`} onClick={onClick} aria-label="PitchIQ home">
    <span className="brand-mark"><Activity size={18} strokeWidth={2.2} /></span>
    <span className="brand-copy"><b>pitch<span>iq</span></b><small>FOOTBALL INTELLIGENCE</small></span>
  </button>;
}

function Eyebrow({ children, light = false }) {
  return <div className={`eyebrow ${light ? 'eyebrow-light' : ''}`}><span className="eyebrow-rule" />{children}</div>;
}

function ArrowLink({ children, onClick, light = false }) {
  return <button className={`arrow-link ${light ? 'arrow-link-light' : ''}`} onClick={onClick}>{children}<ArrowRight size={15} /></button>;
}

function LandingPage({ overview, teams, onEnter }) {
  useReveal([overview, teams]);
  const topTeam = [...(teams || [])].sort((a, b) => Number(b.Overall_Team_Strength) - Number(a.Overall_Team_Strength))[0];
  const blend = (overview?.test_metrics || []).find(item => item.model === overview?.model_metadata?.ensemble_name) || (overview?.test_metrics || []).find(item => item.model === 'PitchIQ v2.5 Ensemble');
  const accuracy = blend ? `${(blend.accuracy * 100).toFixed(1)}%` : '—';
  const logLoss = blend ? number(blend.log_loss, 3) : '—';

  return <div className="landing-page">
    <header className="landing-nav">
      <div className="landing-nav-inner">
        <Brand onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} />
        <nav className="landing-links">
          <a href="#platform">Platform</a>
          <a href="#method">Methodology</a>
          <a href="#evidence">Model evidence</a>
        </nav>
        <button className="nav-cta" onClick={onEnter}>Open workspace <ArrowUpRight size={15} /></button>
      </div>
    </header>

    <main>
      <section className="hero-section">
        <div className="hero-copy">
          <Eyebrow>THE FOOTBALL INTELLIGENCE WORKSPACE</Eyebrow>
          <h1>Read the game.<br /><em>Before it unfolds.</em></h1>
          <p className="hero-lede">A sharper way to explore Premier League form, compare team strength and turn match data into pre-match probabilities.</p>
          <div className="hero-actions">
            <button className="button-dark" onClick={onEnter}>Explore the workspace <ArrowRight size={17} /></button>
            <a className="text-scroll" href="#platform">Discover the platform <ArrowDownRight size={16} /></a>
          </div>
          <div className="hero-meta"><span><i className="pulse-dot" /> 2026/27 DATA SNAPSHOT</span><span>INDEPENDENT FOOTBALL ANALYTICS</span></div>
        </div>
        <div className="hero-visual">
          <div className="hero-photo-wrap">
            <img className="hero-photo" src={HERO_IMAGE} onError={event => { event.currentTarget.style.display = 'none'; }} alt="Footballer accelerating into space during a match" />
            <div className="photo-overlay" />
            <div className="photo-label"><span>01 / MATCH INTELLIGENCE</span><span>THE GAME, IN CONTEXT</span></div>
          </div>
          <div className="float-card float-card-top">
            <span className="float-card-kicker">LEAGUE COVERAGE</span>
            <strong>{overview?.clubs ?? '—'} <small>clubs</small></strong>
            <span className="float-card-foot">2026 / 27 season snapshot</span>
          </div>
          <div className="float-card float-card-bottom">
            <div className="float-card-line"><span className="float-icon"><Gauge size={15} /></span><span>POWER INDEX</span><b>{topTeam ? number(topTeam.Overall_Team_Strength) : '—'}</b></div>
            <div className="float-team-name">{topTeam?.Squad || 'Team ratings'}</div>
            <div className="mini-track"><i style={{ width: `${safePct(topTeam?.Overall_Team_Strength)}%` }} /></div>
          </div>
          <div className="hero-stamp">DATA<br />OVER<br />NOISE<span>✳</span></div>
        </div>
      </section>

      <section className="signal-strip">
        <div className="signal-intro"><span>BUILT AROUND THE DETAILS</span><p>Football is unpredictable.<br /><b>Understanding it shouldn't be.</b></p></div>
        <div className="signal-item"><strong>{overview ? formatNumber(overview.historical_h2h_matches) : '—'}</strong><span>match records<br />available to the model</span></div>
        <div className="signal-item"><strong>{overview ? formatNumber(overview.players) : '—'}</strong><span>player records in<br />the current snapshot</span></div>
        <div className="signal-item"><strong>{overview ? formatNumber(overview.clubs) : '—'}</strong><span>clubs in the<br />current season view</span></div>
      </section>

      <section className="platform-section section-wrap" id="platform">
        <div className="section-heading reveal"><div><Eyebrow>THE PLATFORM / 01</Eyebrow><h2>More than a scoreline.<br /><em>A view of the whole game.</em></h2></div><p>One workspace for forecasts, team profiles and player-level context—built around the data available before a fixture.</p></div>
        <div className="feature-grid">
          <article className="feature-card feature-card-tall reveal">
            <div className="feature-top"><span>01</span><Target size={20} /></div>
            <div className="feature-visual feature-visual-predict">
              <div className="fixture-mini"><div className="fixture-mini-head"><span>MATCH FORECAST</span><span>PRE-MATCH</span></div><div className="fixture-teams"><div><i>ARS</i><b>Arsenal</b></div><span>VS</span><div><i>BHA</i><b>Brighton</b></div></div><div className="fixture-mini-rule"/><div className="fixture-probs"><span>Home win <b>—</b></span><span>Draw <b>—</b></span><span>Away win <b>—</b></span></div></div>
              <div className="orbit orbit-one"/><div className="orbit orbit-two"/>
            </div>
            <h3>Match prediction</h3><p>Explore home, draw and away probabilities alongside expected goals and the model components behind each estimate.</p>
            <ArrowLink onClick={onEnter}>Open match predictor</ArrowLink>
          </article>
          <article className="feature-card reveal">
            <div className="feature-top"><span>02</span><Shield size={20} /></div>
            <div className="feature-visual feature-visual-ratings"><div className="rating-visual-head"><span>CLUB POWER INDEX</span><span>2026/27</span></div>{['Arsenal','Manchester City','Brighton'].map((name, i) => <div className="rating-visual-row" key={name}><i>{String(i + 1).padStart(2, '0')}</i><b>{name}</b><span style={{ width: `${88 - i * 12}%` }} /></div>)}</div>
            <h3>Team intelligence</h3><p>Compare attack, midfield, defence, goalkeeping and squad depth using the current team-strength snapshot.</p>
            <ArrowLink onClick={onEnter}>Explore team ratings</ArrowLink>
          </article>
          <article className="feature-card reveal">
            <div className="feature-top"><span>03</span><Users size={20} /></div>
            <div className="feature-visual feature-visual-players"><div className="player-orbit"><span>FPL</span><i /><b /><em /></div><div className="player-chip chip-a">FWD</div><div className="player-chip chip-b">MID</div><div className="player-chip chip-c">DEF</div></div>
            <h3>Player intelligence</h3><p>Search the current player snapshot by club, position, FPL contribution rating and points.</p>
            <ArrowLink onClick={onEnter}>Explore player data</ArrowLink>
          </article>
        </div>
      </section>

      <section className="editorial-section" id="method">
        <div className="editorial-inner">
          <div className="editorial-image reveal"><img src={MATCH_IMAGE} onError={event => { event.currentTarget.style.display = 'none'; }} alt="Players competing for the ball during a football match" /><span className="image-credit">PHOTO: FRANCO MONSALVO / PEXELS</span><div className="image-caption">THE DETAILS<br /><em>CHANGE THE PICTURE.</em></div></div>
          <div className="editorial-copy reveal"><Eyebrow light>THE METHOD / 02</Eyebrow><h2>Past performance.<br /><em>Current-season signal.</em></h2><p>PitchIQ blends seven statistical and machine-learning models built on Elo, attack/defence ratings and recent form, using only information available before kick-off. Squad ratings and head-to-head context add match-specific detail.</p><div className="method-points"><div><span>01</span><div><b>Historical context</b><small>Every feature is built from earlier results only.</small></div></div><div><span>02</span><div><b>Current-season inputs</b><small>Ratings, form, rest days and expected-goal estimates inform the fixture view.</small></div></div><div><span>03</span><div><b>Transparent output</b><small>Probabilities, expected goals and model components are shown together.</small></div></div></div><button className="button-light" onClick={onEnter}>See the methodology in the workspace <ArrowRight size={16} /></button></div>
        </div>
      </section>

      <section className="evidence-section section-wrap" id="evidence">
        <div className="evidence-heading reveal"><Eyebrow>MODEL EVIDENCE / 03</Eyebrow><h2>Show the work.<br /><em>Not just the prediction.</em></h2><p>Historical holdout performance is available alongside the model. It is evidence about past fixtures—not a guarantee about future matches.</p></div>
        <div className="evidence-card reveal"><div className="evidence-card-head"><span>HISTORICAL HOLDOUT / 2024–26</span><span className="evidence-tag"><i /> BACKTEST</span></div><div className="evidence-number-row"><div><strong>{accuracy}</strong><span>Accuracy · v2.5 ensemble</span></div><div><strong>{logLoss}</strong><span>Log loss · lower is better</span></div></div><div className="evidence-foot"><span>Test sample: {blend ? formatNumber(760) : '—'} fixtures</span><span>Training window: {overview?.training_period || 'Loading data…'}</span></div><div className="evidence-disclaimer">Historical backtest results do not establish 2026/27 prospective performance. Model calibration should be monitored as new match results accrue.</div></div>
      </section>

      <section className="closing-cta"><div><Eyebrow light>YOUR NEXT MATCH STARTS HERE</Eyebrow><h2>See the game<br /><em>from another angle.</em></h2></div><button className="button-light" onClick={onEnter}>Enter PitchIQ <ArrowUpRight size={17} /></button><span className="closing-orbit orbit-a"/><span className="closing-orbit orbit-b"/></section>
    </main>
    <footer className="landing-footer"><Brand onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} /><span>Independent football analytics · Not affiliated with the Premier League or its clubs.</span><a href="https://www.pexels.com/license/" target="_blank" rel="noreferrer">Image licence <ExternalLink size={12} /></a></footer>
  </div>;
}

const NAV_GROUPS = [
  { label: 'WORKSPACE', items: [['Overview', LayoutDashboard], ['Match predictor', Crosshair], ['Matchday predictor', CalendarDays], ['Table predictor', Trophy], ['Team intelligence', Shield], ['Player intelligence', Users]] },
  { label: 'ANALYSIS', items: [['Head to head', Activity], ['Model performance', BarChart3]] },
];

function Sidebar({ active, setActive, onHome, open, onClose }) {
  return <><div className={`sidebar-backdrop ${open ? 'show' : ''}`} onClick={onClose} /><aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
    <div className="sidebar-top"><Brand onClick={onHome} /><button className="icon-button mobile-close" onClick={onClose}><X size={17} /></button></div>
    <button className="competition-switch" onClick={() => setActive('Overview')}><span className="competition-monogram">PL</span><span><b>Premier League</b><small>2026 / 27 season</small></span><ChevronDown size={15} /></button>
    <nav className="side-nav">{NAV_GROUPS.map(group => <div className="nav-group" key={group.label}><div className="nav-group-label">{group.label}</div>{group.items.map(([label, Icon]) => <button key={label} className={`nav-link ${active === label ? 'active' : ''}`} onClick={() => { setActive(label); onClose(); }}><Icon size={17} strokeWidth={1.8} /><span>{label}</span>{active === label && <i />}</button>)}</div>)}</nav>
    <div className="sidebar-bottom"><div className="sidebar-status"><span className="pulse-dot"/><div><b>Data connection</b><small>API snapshot linked</small></div><Check size={14} /></div><div className="sidebar-foot"><span>PITCHIQ / 2.4</span><span>INDEPENDENT PROJECT</span></div></div>
  </aside></>;
}

function Topbar({ active, onMenu, onHome, overview }) {
  return <header className="app-topbar"><button className="icon-button menu-button" onClick={onMenu}><Menu size={18} /></button><div className="app-crumbs"><button onClick={onHome}>PitchIQ</button><ChevronRight size={13} /><b>{active}</b></div><div className="topbar-right"><span className="data-date"><CalendarDays size={14} /> Snapshot · {toDate(overview?.snapshot)}</span><span className="season-chip"><i /> 2026/27</span><div className="user-avatar">PI</div></div></header>;
}

function PageHeader({ eyebrow, title, subtitle, aside }) {
  return <div className="page-header"><div><Eyebrow>{eyebrow}</Eyebrow><h1>{title}</h1><p>{subtitle}</p></div>{aside && <div className="page-header-aside">{aside}</div>}</div>;
}

function StatCard({ label, value, helper, icon: Icon, detail }) {
  return <div className="stat-card"><div className="stat-card-top"><span>{label}</span><Icon size={17} strokeWidth={1.7} /></div><strong>{value}</strong><div className="stat-card-bottom"><span>{helper}</span>{detail && <b>{detail}</b>}</div></div>;
}

function ClubMark({ name, small = false }) {
  return <span className={`club-mark ${small ? 'club-mark-small' : ''}`}>{initials(name)}</span>;
}

function RatingTrack({ value, small = false }) {
  return <span className={`rating-track ${small ? 'rating-track-small' : ''}`}><i style={{ width: `${safePct(value)}%` }} /></span>;
}

function EmptyState({ title, text, action }) {
  return <div className="empty-state"><CircleHelp size={23} /><b>{title}</b><p>{text}</p>{action}</div>;
}

function Dashboard({ overview, teams, onNavigate, onPredict }) {
  const [query, setQuery] = useState('');
  const sorted = useMemo(() => [...teams].sort((a, b) => Number(b.Overall_Team_Strength) - Number(a.Overall_Team_Strength)), [teams]);
  const pointsLeaders = useMemo(() => [...teams].sort((a, b) => Number(b.Pts) - Number(a.Pts)).slice(0, 5), [teams]);
  const filtered = sorted.filter(team => team.Squad?.toLowerCase().includes(query.toLowerCase())).slice(0, 7);
  const recentMetrics = overview?.test_metrics || [];
  const model = recentMetrics.find(item => item.model === overview?.model_metadata?.ensemble_name) || recentMetrics.find(item => item.model === 'PitchIQ v2.5 Ensemble');
  const latest = overview?.snapshot ? toDate(overview.snapshot) : 'Not available';

  return <div className="workspace-page">
    <PageHeader eyebrow="LEAGUE INTELLIGENCE / 01" title="The bigger picture." subtitle="A current-season view of team strength, player data and historical match context." aside={<div className="snapshot-pill"><span className="pulse-dot"/><span><small>LAST DATA SNAPSHOT</small><b>{latest}</b></span></div>} />
    <div className="workspace-hero-banner"><div className="workspace-hero-copy"><span className="banner-kicker">THE 2026 / 27 SEASON</span><h2>Numbers tell a story.<br /><em>Context tells you why.</em></h2><p>Track the league through ratings, form and model-derived match probabilities.</p><button className="button-dark button-small" onClick={() => onNavigate('Match predictor')}>Build a match forecast <ArrowRight size={15}/></button></div><div className="workspace-hero-image"><img src={HERO_IMAGE} onError={event => { event.currentTarget.style.display = 'none'; }} alt="Football player running into space"/><span>FOOTBALL INTELLIGENCE / PITCHIQ</span></div></div>

    <div className="stat-grid">
      <StatCard label="CLUBS TRACKED" value={overview?.clubs ?? '—'} helper="Current league snapshot" icon={Shield} />
      <StatCard label="PLAYER RECORDS" value={formatNumber(overview?.players)} helper="FPL player dataset" icon={Users} />
      <StatCard label="MATCH RECORDS" value={formatNumber(overview?.historical_h2h_matches)} helper="Model context records" icon={Activity} />
      <StatCard label="TOP TEAM INDEX" value={number(overview?.top_rating)} helper={overview?.highest_rated_team || 'Awaiting data'} icon={Gauge} detail="/ 100" />
    </div>

    <div className="dashboard-grid">
      <section className="surface-panel ranking-panel">
        <div className="panel-heading"><div><span className="panel-kicker">POWER INDEX / 2026–27</span><h2>Team strength</h2><p>Model-derived ratings across current-season team inputs.</p></div><ArrowLink onClick={() => onNavigate('Team intelligence')}>All clubs</ArrowLink></div>
        <div className="table-toolbar"><span>{teams.length} clubs · overall index</span><label className="table-search"><Search size={14}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search clubs" /></label></div>
        <div className="team-table"><div className="team-table-head"><span>CLUB</span><span>STRENGTH INDEX</span><span>SCORE</span></div>
          {filtered.map(team => <button className="team-table-row" key={team.Squad} onClick={() => onNavigate('Team intelligence')}><span className="team-cell"><i>{String(sorted.findIndex(t => t.Squad === team.Squad) + 1).padStart(2, '0')}</i><ClubMark name={team.Squad}/><b>{team.Squad}</b></span><RatingTrack value={team.Overall_Team_Strength}/><strong>{number(team.Overall_Team_Strength)}</strong></button>)}
          {!filtered.length && <div className="table-empty">No clubs match that search.</div>}
        </div><div className="table-footnote">Ratings are independent model indices, not official league standings.</div>
      </section>
      <div className="dashboard-side-stack">
        <section className="surface-panel leaders-panel"><div className="panel-heading"><div><span className="panel-kicker">LEAGUE PULSE</span><h2>Points leaders</h2></div><Trophy size={18}/></div><div className="leaders-list">{pointsLeaders.map((team, index) => <div className="leader-row" key={team.Squad}><i>{String(index + 1).padStart(2, '0')}</i><ClubMark name={team.Squad} small/><b>{team.Squad}</b><strong>{team.Pts}<small> pts</small></strong></div>)}</div><div className="table-footnote">From the supplied league table snapshot.</div></section>
        <section className="model-summary-card"><div className="model-summary-top"><span><Sparkles size={15}/> MODEL NOTE</span><span className="live-label"><i/> ACTIVE</span></div><h3>Ratings carry the signal.<br/><em>Seven models, one blend.</em></h3><p>{overview?.model_type || 'Hybrid prediction model'} combines Elo, attack/defence ratings and Dixon-Coles with machine-learning classifiers, all built from strictly pre-match results.</p><div className="model-summary-bottom"><span>HISTORICAL WINDOW</span><b>{overview?.training_period || '—'}</b></div></section>
      </div>
    </div>
    <div className="dashboard-bottom-grid">
      <section className="surface-panel forecast-teaser"><div><span className="panel-kicker">MATCH CENTRE</span><h2>Start with a fixture.</h2><p>Compare two clubs and inspect the probability split, expected goals and rating inputs.</p></div><button className="button-dark" onClick={() => onNavigate('Match predictor')}>Open predictor <ArrowRight size={16}/></button></section>
      <section className="surface-panel evidence-teaser"><div className="evidence-teaser-icon"><BarChart3 size={19}/></div><div><span className="panel-kicker">MODEL PERFORMANCE</span><h2>{model ? `${(model.accuracy * 100).toFixed(1)}%` : '—'} historical accuracy</h2><p>v2.5 ensemble · 2024/25–2025/26 holdout</p></div><button className="round-arrow" onClick={() => onNavigate('Model performance')} aria-label="Open model performance"><ArrowUpRight size={17}/></button></section>
    </div>
    <PageFooter />
  </div>;
}

function ProbabilityBar({ label, value, tone }) {
  return <div className="probability-row"><span>{label}</span><div className="probability-track"><i className={tone} style={{ width: `${safePct(value)}%` }} /></div><b>{number(value, 1)}%</b></div>;
}

function PredictorPage({ teams }) {
  const names = useMemo(() => teams.map(team => team.Squad).sort((a, b) => a.localeCompare(b)), [teams]);
  const [home, setHome] = useState('Arsenal');
  const [away, setAway] = useState('Manchester City');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  useEffect(() => { if (names.length && !names.includes(home)) setHome(names[0]); if (names.length && !names.includes(away)) setAway(names[Math.min(1, names.length - 1)]); }, [names]);

  async function runPrediction() {
    if (!home || !away || home === away) { setError('Choose two different clubs to run a forecast.'); return; }
    setLoading(true); setError('');
    try { setPrediction(await getJSON('/api/predict', { method: 'POST', body: JSON.stringify({ home_team: home, away_team: away }) })); }
    catch (err) { setError(err.message); setPrediction(null); }
    finally { setLoading(false); }
  }
  const homeRatings = prediction?.team_ratings?.home;
  const awayRatings = prediction?.team_ratings?.away;
  const ratingLabels = [['Overall', 'overall'], ['Attack', 'attack'], ['Midfield', 'midfield'], ['Defence', 'defence'], ['Goalkeeping', 'goalkeeping'], ['Squad depth', 'depth']];

  return <div className="workspace-page">
    <PageHeader eyebrow="MATCH CENTRE / 02" title="Make the comparison." subtitle="Generate a pre-match estimate and inspect the signals behind it." aside={<span className="method-chip"><Activity size={14}/> Hybrid model</span>} />
    <div className="predictor-layout">
      <section className="surface-panel fixture-builder">
        <div className="panel-heading"><div><span className="panel-kicker">FIXTURE SETUP</span><h2>Choose your matchup</h2><p>Venue is reflected in the home/away model inputs.</p></div><span className="builder-icon"><Crosshair size={20}/></span></div>
        <div className="fixture-builder-image"><img src={MATCH_IMAGE} onError={event => { event.currentTarget.style.display = 'none'; }} alt="Two footballers competing for the ball"/><div><span>THE NEXT 90 MINUTES</span><b>Start with the data.</b></div></div>
        <div className="club-select-grid">
          <label className="club-select"><span>HOME CLUB</span><div><ClubMark name={home}/><select value={home} onChange={e => setHome(e.target.value)}>{names.map(name => <option key={name}>{name}</option>)}</select><ChevronDown size={15}/></div></label>
          <div className="versus-mark">VS</div>
          <label className="club-select"><span>AWAY CLUB</span><div><ClubMark name={away}/><select value={away} onChange={e => setAway(e.target.value)}>{names.map(name => <option key={name}>{name}</option>)}</select><ChevronDown size={15}/></div></label>
        </div>
        <button className="button-dark predict-button" onClick={runPrediction} disabled={loading || !teams.length}>{loading ? <><RefreshCw size={16} className="spin"/> Calculating estimate…</> : <>Generate match estimate <ArrowRight size={16}/></>}</button>
        {error && <p className="form-error">{error}</p>}
        <div className="builder-disclaimer"><CircleHelp size={14}/> Forecasts are estimates based on available pre-match inputs. They are not guarantees.</div>
      </section>
      <section className="surface-panel prediction-results">
        {!prediction ? <div className="prediction-empty"><div className="empty-orbit"><Target size={30}/></div><span className="panel-kicker">PREDICTION OUTPUT</span><h2>Your match view<br/><em>starts here.</em></h2><p>Select the two clubs and generate an estimate. Probabilities and expected goals will appear in this panel.</p><div className="empty-result-note"><span><i/> READY WHEN YOU ARE</span><span>HOME · DRAW · AWAY</span></div></div> : <>
          <div className="result-header"><div><span className="panel-kicker">MODEL PROBABILITIES</span><h2>{prediction.predicted_outcome}</h2><p>{prediction.home_team} vs {prediction.away_team}</p></div><span className="result-status"><i/> ESTIMATE READY</span></div>
          <div className="probability-total"><div className="probability-primary"><span>Most likely outcome</span><strong>{prediction.predicted_outcome}</strong></div><div className="probability-total-value">{number(Math.max(...Object.values(prediction.probabilities)), 1)}<small>%</small></div></div>
          <div className="probability-list"><ProbabilityBar label="Home win" value={prediction.probabilities.home_win} tone="tone-home"/><ProbabilityBar label="Draw" value={prediction.probabilities.draw} tone="tone-draw"/><ProbabilityBar label="Away win" value={prediction.probabilities.away_win} tone="tone-away"/></div>
          <div className="expected-goals"><div><span>EXPECTED GOALS</span><strong>{number(prediction.expected_goals.home, 2)}</strong><small>{prediction.home_team}</small></div><div className="xg-divider">—</div><div><span>EXPECTED GOALS</span><strong>{number(prediction.expected_goals.away, 2)}</strong><small>{prediction.away_team}</small></div></div>
          <div className="result-footnote">The displayed probabilities are the equal-weight blend of seven models; no manual home-advantage or draw adjustment is applied.</div>
        </>}
      </section>
    </div>
    {prediction && <>
      <section className="surface-panel component-panel"><div className="panel-heading"><div><span className="panel-kicker">TEAM COMPARISON</span><h2>Where the matchup differs</h2><p>Current-season team-strength components supplied to the prediction engine.</p></div></div>
        <div className="comparison-head"><div><ClubMark name={prediction.home_team}/><b>{prediction.home_team}</b></div><span>COMPONENT INDEX / 100</span><div><ClubMark name={prediction.away_team}/><b>{prediction.away_team}</b></div></div>
        {ratingLabels.map(([label, key]) => <div className="comparison-row" key={key}><div className="comparison-value">{number(homeRatings?.[key])}</div><div className="comparison-center"><span>{label}</span><div className="comparison-tracks"><RatingTrack value={homeRatings?.[key]}/><RatingTrack value={awayRatings?.[key]}/></div></div><div className="comparison-value align-right">{number(awayRatings?.[key])}</div></div>)}
      </section>
      <section className="surface-panel model-breakdown"><div className="panel-heading"><div><span className="panel-kicker">MODEL COMPOSITION</span><h2>What went into the estimate</h2></div><span className="method-chip">{prediction.model}</span></div><div className="blend-grid"><div><span>CLASSIFIER MODELS</span><strong>{prediction.model_components?.historical_classifier_weight_pct ?? '—'}%</strong><small>Elo, logistic, forest, XGBoost</small></div><div><span>GOAL MODELS</span><strong>{prediction.model_components?.current_season_poisson_weight_pct ?? '—'}%</strong><small>Home/away Poisson and Dixon-Coles</small></div><div><span>HOME ADVANTAGE</span><strong>Learned</strong><small>Estimated from results, no manual factor</small></div></div><p className="model-caveat">{prediction.model_disclaimer}</p>
      </section>
    </>}
    <PageFooter />
  </div>;
}

function TeamIntelligencePage({ teams }) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const sorted = useMemo(() => [...teams].sort((a, b) => Number(b.Overall_Team_Strength) - Number(a.Overall_Team_Strength)), [teams]);
  const filtered = sorted.filter(team => team.Squad?.toLowerCase().includes(query.toLowerCase()));
  useEffect(() => {
    let cancelled = false;
    if (!selected) { setDetail(null); return; }
    setLoading(true); setDetail(null);
    getJSON(`/api/teams/${encodeURIComponent(selected)}`).then(data => { if (!cancelled) setDetail(data); }).catch(err => { if (!cancelled) setDetail({ error: err.message }); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [selected]);
  const fields = [['Attack', 'Attacking_Strength'], ['Midfield', 'Midfield_Strength'], ['Defence', 'Defensive_Strength'], ['Goalkeeping', 'Goalkeeping_Strength'], ['Squad depth', 'Squad_Depth_Score']];
  return <div className="workspace-page">
    <PageHeader eyebrow="TEAM INTELLIGENCE / 03" title="The shape of a team." subtitle="Explore independent club ratings and the components that make up each team's profile." aside={<span className="method-chip"><Shield size={14}/> Current-season snapshot</span>} />
    <div className="team-intel-layout">
      <section className="surface-panel team-directory"><div className="panel-heading"><div><span className="panel-kicker">CLUB DIRECTORY</span><h2>All clubs</h2><p>Sorted by overall team-strength index.</p></div></div><label className="wide-search"><Search size={15}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search a club"/><span>{filtered.length} clubs</span></label>
        <div className="club-list">{filtered.map((team, index) => <button className={`club-list-row ${selected === team.Squad ? 'selected' : ''}`} key={team.Squad} onClick={() => setSelected(team.Squad)}><span className="club-list-rank">{String(sorted.findIndex(t => t.Squad === team.Squad) + 1).padStart(2, '0')}</span><ClubMark name={team.Squad}/><span className="club-list-info"><b>{team.Squad}</b><small>{team.MP ?? '—'} matches · {team.Pts ?? '—'} points</small></span><span className="club-list-rating">{number(team.Overall_Team_Strength)}</span><ChevronRight size={15}/></button>)}{!filtered.length && <div className="table-empty">No clubs found.</div>}</div>
      </section>
      <section className="surface-panel team-detail-panel">{!selected ? <div className="prediction-empty"><div className="empty-orbit"><Shield size={28}/></div><span className="panel-kicker">CLUB PROFILE</span><h2>Select a club<br/><em>to inspect its profile.</em></h2><p>Choose a club from the directory to view its strength breakdown, league snapshot and squad overview.</p></div> : loading ? <div className="detail-loading"><span className="spin-loader"/><b>Loading club profile…</b></div> : detail?.error ? <EmptyState title="Could not load this profile" text={detail.error}/> : detail?.team ? <>
        <div className="team-detail-head"><div><span className="panel-kicker">TEAM PROFILE / 2026–27</span><div className="team-detail-name"><ClubMark name={selected}/><h2>{selected}</h2></div><p>Independent current-season strength profile.</p></div><div className="overall-score"><strong>{number(detail.team.Overall_Team_Strength)}</strong><span>OVERALL / 100</span></div></div>
        <div className="team-component-list">{fields.map(([label, key]) => <div className="team-component" key={key}><span>{label}</span><RatingTrack value={detail.team[key]}/><b>{number(detail.team[key])}</b></div>)}</div>
        <div className="team-detail-stats"><div><span>LEAGUE POSITION</span><strong>{detail.team.Rk ?? '—'}</strong></div><div><span>PLAYED</span><strong>{detail.team.MP ?? '—'}</strong></div><div><span>POINTS</span><strong>{detail.team.Pts ?? '—'}</strong></div><div><span>GOAL DIFFERENCE</span><strong>{detail.team.GD > 0 ? '+' : ''}{detail.team.GD ?? '—'}</strong></div></div>
        <div className="squad-overview"><div className="panel-kicker">SQUAD SNAPSHOT</div><div><span>Starting XI FPL rating</span><b>{number(detail.team.Starting_XI_FPL_Rating)}</b></div><div><span>Bench rating</span><b>{number(detail.team.Bench_FPL_Rating)}</b></div><div><span>Players in dataset</span><b>{detail.players?.length ?? '—'}</b></div></div>
      </> : <div className="prediction-empty"><h2>Choose a club.</h2></div>}</section>
    </div>
    <PageFooter />
  </div>;
}

function PlayerIntelligencePage({ teams }) {
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [club, setClub] = useState('All clubs');
  const [position, setPosition] = useState('All positions');
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true); setError('');
      try {
        const all = await Promise.all(teams.map(team => getJSON(`/api/teams/${encodeURIComponent(team.Squad)}`)));
        const roster = all.flatMap(data => (data.players || []).map(player => ({ ...player, Club: player.Club || data.team?.Squad })));
        if (!cancelled) setPlayers(roster);
      } catch (err) { if (!cancelled) setError(err.message); }
      finally { if (!cancelled) setLoading(false); }
    }
    if (teams.length) load(); else { setLoading(false); setError('Team data is not available.'); }
    return () => { cancelled = true; };
  }, [teams]);
  const filtered = useMemo(() => players.filter(player => {
    const matchesQuery = `${player.player_name || ''} ${player.Club || ''}`.toLowerCase().includes(query.toLowerCase());
    const matchesClub = club === 'All clubs' || player.Club === club;
    const matchesPosition = position === 'All positions' || player.Position === position;
    return matchesQuery && matchesClub && matchesPosition;
  }).sort((a, b) => Number(b.FPL_Contribution_Rating || 0) - Number(a.FPL_Contribution_Rating || 0)), [players, query, club, position]);
  const topPlayer = filtered[0];
  return <div className="workspace-page">
    <PageHeader eyebrow="PLAYER INTELLIGENCE / 04" title="The people behind the numbers." subtitle="Explore the current FPL player snapshot by club, position, points and contribution rating." aside={<span className="method-chip"><Users size={14}/> Player snapshot</span>} />
    <div className="player-toolbar surface-panel"><label className="wide-search"><Search size={15}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search players or clubs"/><span>{loading ? 'Loading…' : `${formatNumber(filtered.length)} players`}</span></label><select value={club} onChange={e => setClub(e.target.value)}><option>All clubs</option>{teams.map(team => <option key={team.Squad}>{team.Squad}</option>)}</select><select value={position} onChange={e => setPosition(e.target.value)}><option>All positions</option><option>GK</option><option>DEF</option><option>MID</option><option>FWD</option></select></div>
    {error && <div className="inline-error-panel">{error}</div>}
    {loading ? <div className="surface-panel loading-panel"><span className="spin-loader"/><b>Collecting player snapshots from club profiles…</b><p>This page reads the existing team-detail endpoints; no player metrics are fabricated.</p></div> : <>
      {topPlayer && <section className="player-feature surface-panel"><div className="player-feature-symbol"><Users size={22}/></div><div><span className="panel-kicker">TOP CONTRIBUTION RATING / FILTERED VIEW</span><h2>{topPlayer.player_name}</h2><p>{topPlayer.Club} · {topPlayer.Position} · {topPlayer.total_points ?? 0} total points</p></div><div className="player-feature-score"><strong>{number(topPlayer.FPL_Contribution_Rating)}</strong><span>CONTRIBUTION INDEX</span></div></section>}
      <section className="surface-panel player-table-panel"><div className="panel-heading"><div><span className="panel-kicker">PLAYER DIRECTORY</span><h2>Current player records</h2><p>Use the filters to narrow the current-season snapshot.</p></div></div><div className="player-table"><div className="player-table-head"><span>PLAYER</span><span>CLUB / POSITION</span><span>FPL POINTS</span><span>CONTRIBUTION</span><span>PRICE</span></div>{filtered.slice(0, 100).map(player => <div className="player-table-row" key={`${player.Player_ID}-${player.Club}`}><div className="player-name-cell"><span className="player-initials">{initials(player.player_name)}</span><b>{player.player_name}</b></div><div className="player-club-cell"><b>{player.Club}</b><small>{player.Position}</small></div><strong>{player.total_points ?? '—'}</strong><strong>{number(player.FPL_Contribution_Rating)}</strong><span>£{number(player.now_cost, 1)}m</span></div>)}{!filtered.length && <div className="table-empty">No players match the selected filters.</div>}</div>{filtered.length > 100 && <div className="table-footnote">Showing the first 100 matching players. Refine your search to narrow the list.</div>}</section>
    </>}
    <PageFooter />
  </div>;
}

function HeadToHeadPage({ teams }) {
  const names = useMemo(() => teams.map(team => team.Squad).sort((a, b) => a.localeCompare(b)), [teams]);
  const [home, setHome] = useState('Arsenal');
  const [away, setAway] = useState('Manchester City');
  const [limit, setLimit] = useState('10');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { if (names.length) { if (!names.includes(home)) setHome(names[0]); if (!names.includes(away)) setAway(names[Math.min(1, names.length - 1)]); } }, [names]);
  async function compare() {
    if (!home || !away || home === away) { setError('Choose two different clubs.'); return; }
    setLoading(true); setError('');
    try { const params = new URLSearchParams({ home_team: home, away_team: away, limit }); setData(await getJSON(`/api/h2h?${params.toString()}`)); }
    catch (err) { setError(err.message); setData(null); }
    finally { setLoading(false); }
  }
  return <div className="workspace-page">
    <PageHeader eyebrow="HEAD-TO-HEAD / 05" title="History between the lines." subtitle="Review previous meetings as supporting context—not as a standalone forecast." aside={<span className="method-chip"><Activity size={14}/> Historical context</span>} />
    <section className="surface-panel h2h-controls"><div className="panel-heading"><div><span className="panel-kicker">HEAD-TO-HEAD SELECTOR</span><h2>Choose two clubs</h2><p>Historical meetings are listed from newest to oldest.</p></div></div><div className="h2h-selects"><label><span>CLUB ONE</span><select value={home} onChange={e => setHome(e.target.value)}>{names.map(name => <option key={name}>{name}</option>)}</select></label><span className="versus-mark">VS</span><label><span>CLUB TWO</span><select value={away} onChange={e => setAway(e.target.value)}>{names.map(name => <option key={name}>{name}</option>)}</select></label><label className="limit-select"><span>MEETINGS</span><select value={limit} onChange={e => setLimit(e.target.value)}><option value="5">Last 5</option><option value="10">Last 10</option><option value="20">Last 20</option><option value="30">Last 30</option></select></label><button className="button-dark" onClick={compare} disabled={loading}>{loading ? 'Loading…' : 'Compare meetings'} <ArrowRight size={15}/></button></div>{error && <p className="form-error">{error}</p>}</section>
    {data ? <>
      <div className="h2h-stat-grid"><StatCard label={`${data.home_team} WINS`} value={data.home_team_wins} helper="From selected club's perspective" icon={ArrowUpRight}/><StatCard label="DRAWS" value={data.draws} helper="Level after 90 minutes" icon={Activity}/><StatCard label={`${data.away_team} WINS`} value={data.away_team_wins} helper="From selected club's perspective" icon={ArrowDownRight}/><StatCard label="MEETINGS FOUND" value={data.matches_found} helper={`Requested: ${data.matches_requested}`} icon={CalendarDays}/></div>
      <section className="surface-panel h2h-list-panel"><div className="panel-heading"><div><span className="panel-kicker">PREVIOUS MEETINGS</span><h2>{data.home_team} / {data.away_team}</h2><p>{data.note}</p></div></div><div className="h2h-list">{data.matches.map((match, index) => <div className="h2h-row" key={`${match.date}-${index}`}><span className="h2h-date">{toDate(match.date)}</span><span className="h2h-teams"><b>{match.home_team}</b><span>{match.score}</span><b>{match.away_team}</b></span><span className={`h2h-result result-${match.home_team_perspective_result.toLowerCase()}`}>{match.home_team_perspective_result}</span><span className="h2h-season">{match.season}</span></div>)}{!data.matches.length && <div className="table-empty">No previous meetings were found in the available archive.</div>}</div></section>
    </> : <section className="surface-panel h2h-empty"><div className="empty-orbit"><Activity size={27}/></div><h2>Look back before you look ahead.</h2><p>Choose two clubs to load the available head-to-head archive.</p></section>}
    <PageFooter />
  </div>;
}

function ModelPerformancePage({ overview }) {
  const metrics = overview?.test_metrics || [];
  const blend = metrics.find(item => item.model === overview?.model_metadata?.ensemble_name) || metrics.find(item => item.model === 'PitchIQ v2.5 Ensemble');
  const sortedByLog = [...metrics].filter(item => item.log_loss != null).sort((a, b) => a.log_loss - b.log_loss);
  return <div className="workspace-page">
    <PageHeader eyebrow="MODEL PERFORMANCE / 06" title="Evidence, not promises." subtitle="Inspect the model's historical holdout results and the blend used for current predictions." aside={<span className="method-chip"><BarChart3 size={14}/> Backtest report</span>} />
    <div className="performance-callout"><div><span className="panel-kicker">TEST PERIOD / 2024–25 TO 2025–26</span><h2>Historical holdout performance</h2><p>The test set is separate from the training window. Results describe past fixtures and should not be interpreted as a guarantee of future prediction quality.</p></div><div className="performance-callout-meta"><strong>{blend ? formatNumber(overview?.model_metadata?.test_matches ?? 760) : '—'}</strong><span>TEST FIXTURES</span></div></div>
    <div className="performance-stat-grid"><StatCard label="BLENDED ACCURACY" value={blend ? `${(blend.accuracy * 100).toFixed(1)}%` : '—'} helper={overview?.model_metadata?.ensemble_name || 'Selected classifier + Poisson'} icon={Target}/><StatCard label="BLENDED LOG LOSS" value={blend ? number(blend.log_loss, 3) : '—'} helper="Lower is better" icon={Activity}/><StatCard label="BLENDED BRIER SCORE" value={blend ? number(blend.brier_score, 3) : '—'} helper="Lower is better" icon={Gauge}/><StatCard label="TRAINING RECORDS" value={formatNumber(overview?.training_matches)} helper={overview?.training_period || '—'} icon={Database}/></div>
    <section className="surface-panel class-recall-panel"><div className="panel-heading"><div><span className="panel-kicker">CLASS-LEVEL DIAGNOSTICS</span><h2>Which outcomes does the model recognise?</h2><p>Recall measures the share of actual outcomes correctly identified as each class.</p></div></div><div className="class-recall-grid"><div><span>HOME WIN RECALL</span><b>{blend ? `${(blend.home_recall * 100).toFixed(1)}%` : '—'}</b></div><div className="draw-recall-warning"><span>DRAW RECALL</span><b>{blend ? `${(blend.draw_recall * 100).toFixed(1)}%` : '—'}</b><small>Known weakness; draw boosts degraded validation probability scores.</small></div><div><span>AWAY WIN RECALL</span><b>{blend ? `${(blend.away_recall * 100).toFixed(1)}%` : '—'}</b></div></div></section>
    {(() => { const v24 = metrics.find(i => i.model?.startsWith('v2.4 RF+Poisson (as shipped')); return blend && v24 ? <p className="model-caveat">On the same {formatNumber(overview?.model_metadata?.test_matches ?? 760)} fixtures, v2.5 log loss is {number(blend.log_loss, 3)} against {number(v24.log_loss, 3)} for v2.4 (lower is better). The gap is small relative to sampling noise, and draws are not separable from the available data, so the model almost never picks a draw as the single most likely result.</p> : null; })()}
    <section className="surface-panel model-results-panel"><div className="panel-heading"><div><span className="panel-kicker">MODEL COMPARISON</span><h2>Holdout results by model</h2><p>Baselines, the v2.4 model and every v2.5 component on the same fixtures.</p></div></div><div className="model-result-head"><span>MODEL</span><span>ACCURACY</span><span>LOG LOSS</span><span>BRIER SCORE</span></div>{metrics.map(item => <div className={`model-result-row ${item.model === overview?.model_metadata?.ensemble_name ? 'model-result-highlight' : ''}`} key={item.model}><div><b>{item.model}</b>{item.model === overview?.model_metadata?.ensemble_name && <small>Selected blend</small>}</div><strong>{(item.accuracy * 100).toFixed(1)}%</strong><strong>{number(item.log_loss, 3)}</strong><strong>{number(item.brier_score, 3)}</strong></div>)}{!metrics.length && <div className="table-empty">Model test metrics are not available from the API.</div>}<div className="table-footnote">Accuracy = correct class rate. Log loss and Brier score evaluate probability quality; lower values are better.</div></section>
    <section className="surface-panel model-method-panel"><div className="panel-heading"><div><span className="panel-kicker">PREDICTION PIPELINE</span><h2>How the forecast is assembled</h2></div></div><div className="pipeline-steps"><div><span>01</span><b>Ratings and features</b><p>Elo, attack/defence ratings, opponent-adjusted form, rest days and head-to-head, built only from earlier results.</p></div><i><ArrowRight size={15}/></i><div><span>02</span><b>Seven models</b><p>Elo, two logistic models, random forest, XGBoost, home/away Poisson goals and Dixon-Coles.</p></div><i><ArrowRight size={15}/></i><div><span>03</span><b>Equal-weight blend</b><p>{number((overview?.model_metadata?.ml_probability_weight ?? 0) * 100, 0)}% classifiers / {number((overview?.model_metadata?.poisson_probability_weight ?? 0) * 100, 0)}% goal models. No draw or home-advantage adjustment is applied.</p></div></div><p className="model-caveat">{overview?.training_status || 'Training metadata is not available.'}</p></section>
    <PageFooter />
  </div>;
}


function MatchdayFixtureCard({ item }) {
  const prediction = item.prediction;
  const probabilities = prediction.probabilities;
  return <article className="surface-panel matchday-fixture-card">
    <div className="matchday-fixture-top"><span>FIXTURE {String(item.fixture_number).padStart(2, '0')}</span><span className="prediction-outcome">{prediction.predicted_outcome}</span></div>
    <div className="matchday-fixture-teams">
      <div className="matchday-club"><ClubMark name={item.home_team}/><b>{item.home_team}</b><small>HOME</small></div>
      <div className="matchday-score"><strong>{number(prediction.expected_goals.home, 2)}</strong><span>EXPECTED GOALS</span><strong>{number(prediction.expected_goals.away, 2)}</strong></div>
      <div className="matchday-club away"><ClubMark name={item.away_team}/><b>{item.away_team}</b><small>AWAY</small></div>
    </div>
    <div className="matchday-probabilities">
      <div><span>HOME</span><b>{number(probabilities.home_win, 1)}%</b><i><em className="home" style={{width:`${safePct(probabilities.home_win)}%`}}/></i></div>
      <div><span>DRAW</span><b>{number(probabilities.draw, 1)}%</b><i><em className="draw" style={{width:`${safePct(probabilities.draw)}%`}}/></i></div>
      <div><span>AWAY</span><b>{number(probabilities.away_win, 1)}%</b><i><em className="away" style={{width:`${safePct(probabilities.away_win)}%`}}/></i></div>
    </div>
    <div className="matchday-card-foot"><span>Most likely outcome</span><b>{item.confidence_pct.toFixed(1)}% probability</b><span className="model-caption">{prediction.model}</span></div>
  </article>;
}

function MatchdayPredictorPage() {
  const [matchday, setMatchday] = useState(11);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoading(true); setError(''); setData(null);
      try { const result = await getJSON(`/api/matchday/${matchday}`); if (mounted) setData(result); }
      catch (err) { if (mounted) setError(err.message); }
      finally { if (mounted) setLoading(false); }
    }
    load();
    return () => { mounted = false; };
  }, [matchday]);
  const rounds = Array.from({length: 28}, (_, i) => i + 11);
  return <div className="workspace-page">
    <PageHeader eyebrow="MATCHDAY FORECAST / 07" title="The full round, one forecast." subtitle="Select a matchday to generate probabilities for every outstanding fixture using the same PitchIQ prediction engine." aside={<label className="matchday-selector"><span>SELECT MATCHDAY</span><select value={matchday} onChange={e => setMatchday(Number(e.target.value))}>{rounds.map(round => <option value={round} key={round}>Matchday {round}</option>)}</select></label>} />
    {loading && <section className="surface-panel predictor-loading"><RefreshCw className="spin" size={20}/><b>Building Matchday {matchday} forecasts</b><span>Scoring all fixtures with the current PitchIQ model…</span></section>}
    {error && <section className="surface-panel predictor-error"><b>Fixtures could not be forecast.</b><p>{error}</p></section>}
    {data && <>
      <div className="matchday-summary-grid">
        <StatCard label="FIXTURES" value={data.fixtures_count} helper={`Matchday ${data.matchday}`} icon={CalendarDays}/>
        <StatCard label="HOME WINS" value={data.summary.home_wins} helper="Highest probability outcome" icon={Target}/>
        <StatCard label="DRAWS" value={data.summary.draws} helper="Highest probability outcome" icon={Activity}/>
        <StatCard label="AWAY WINS" value={data.summary.away_wins} helper="Highest probability outcome" icon={ArrowUpRight}/>
      </div>
      <div className="matchday-insight-grid">
        <section className="surface-panel matchday-insight"><span className="panel-kicker">HIGHEST CONFIDENCE</span><b>{data.most_confident_fixture.home_team} vs {data.most_confident_fixture.away_team}</b><strong>{number(data.most_confident_fixture.confidence_pct, 1)}%</strong><span>{data.most_confident_fixture.outcome}</span></section>
        <section className="surface-panel matchday-insight"><span className="panel-kicker">CLOSEST FIXTURE</span><b>{data.closest_fixture.home_team} vs {data.closest_fixture.away_team}</b><strong>{number(data.closest_fixture.probability_margin_pct, 1)} pp</strong><span>Gap between the top two outcome probabilities</span></section>
      </div>
      <div className="matchday-section-heading"><div><span className="panel-kicker">MATCHDAY {data.matchday} / ALL FIXTURES</span><h2>Fixture-by-fixture forecast</h2></div><span className="method-chip"><Activity size={14}/> {data.model}</span></div>
      <div className="matchday-fixture-grid">{data.fixtures.map(item => <MatchdayFixtureCard key={`${item.home_team}-${item.away_team}`} item={item}/>)}</div>
      <p className="model-caveat">Probabilities represent model estimates, not certainties. Matchday fixture dates and kick-off times are not included in the current schedule snapshot.</p>
    </>}
    <PageFooter />
  </div>;
}

function TablePredictorPage() {
  const [simulations, setSimulations] = useState(10000);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  async function runSimulation() {
    setLoading(true); setError('');
    try { const result = await getJSON(`/api/table-prediction?simulations=${simulations}`); setData(result); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }
  const titleLeader = data?.table?.reduce((best, row) => !best || row.title_probability > best.title_probability ? row : best, null);
  const topFourLeader = data?.table?.reduce((best, row) => !best || row.top4_probability > best.top4_probability ? row : best, null);
  const relegationRisk = data?.table?.reduce((best, row) => !best || row.relegation_probability > best.relegation_probability ? row : best, null);
  return <div className="workspace-page">
    <PageHeader eyebrow="SEASON OUTLOOK / 08" title="Where could the table finish?" subtitle="Simulate the remaining 2026/27 fixtures to estimate final points, goal difference and each club's finishing-position probabilities." aside={<div className="table-sim-controls"><label><span>SIMULATION RUNS</span><select value={simulations} onChange={e => setSimulations(Number(e.target.value))}><option value={10000}>10,000</option><option value={20000}>20,000</option><option value={50000}>50,000</option></select></label><button className="button-dark" onClick={runSimulation} disabled={loading}><RefreshCw size={14} className={loading ? 'spin' : ''}/>{loading ? 'Simulating…' : data ? 'Run again' : 'Run simulation'}</button></div>} />
    <section className="surface-panel table-method-banner"><div className="table-method-icon"><Trophy size={20}/></div><div><b>Monte Carlo season simulation</b><p>Every remaining fixture is forecast by PitchIQ, then scorelines are sampled across repeated season runs. The projected table includes goal difference, not just points.</p></div></section>
    {error && <section className="surface-panel predictor-error"><b>Simulation could not be completed.</b><p>{error}</p></section>}
    {loading && <section className="surface-panel predictor-loading"><RefreshCw className="spin" size={20}/><b>Simulating {formatNumber(simulations)} possible seasons</b><span>Processing all remaining fixtures and ranking the final tables…</span></section>}
    {!data && !loading && <section className="surface-panel table-empty-state"><div className="empty-orbit"><Trophy size={26}/></div><h2>Run the rest of the season.</h2><p>PitchIQ will simulate all 280 remaining fixtures and estimate each club's final position, points and key league-outcome probabilities.</p><button className="button-dark" onClick={runSimulation}><Trophy size={15}/> Simulate 2026/27 season</button></section>}
    {data && <>
      <div className="table-sim-summary"><div className="table-sim-summary-main"><span className="panel-kicker">SIMULATION COMPLETE</span><strong>{formatNumber(data.simulations)}</strong><span>season runs · {data.fixtures_simulated} remaining fixtures per run</span></div><div className="table-sim-summary-side"><span>MODEL</span><b>{data.model}</b><span>Standings after Matchday {data.standings_through_matchday}</span></div></div>
      <div className="table-outcome-cards">
        <StatCard label="TITLE FAVOURITE" value={titleLeader?.team || '—'} helper={`${number(titleLeader?.title_probability, 1)}% title probability`} icon={Trophy}/>
        <StatCard label="TOP-FOUR FAVOURITE" value={topFourLeader?.team || '—'} helper={`${number(topFourLeader?.top4_probability, 1)}% top-four probability`} icon={Target}/>
        <StatCard label="HIGHEST RELEGATION RISK" value={relegationRisk?.team || '—'} helper={`${number(relegationRisk?.relegation_probability, 1)}% bottom-three probability`} icon={Activity}/>
      </div>
      <section className="surface-panel projected-table-panel"><div className="panel-heading"><div><span className="panel-kicker">PROJECTED FINAL STANDINGS / 2026–27</span><h2>End-of-season table</h2><p>Ordered by expected points, goal difference and goals scored. Points and records are simulation averages.</p></div><span className="method-chip"><Trophy size={14}/> 38 matches</span></div>
        <div className="projected-table-scroll"><table className="projected-table"><thead><tr><th>POS</th><th>CLUB</th><th>MP</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>PTS</th><th>TITLE</th><th>TOP 4</th><th>RELEG.</th></tr></thead><tbody>{data.table.map(row => <tr key={row.team} className={row.projected_position <= 4 ? 'projected-top-four' : row.projected_position >= 18 ? 'projected-relegation' : ''}><td className="position-cell">{row.projected_position}</td><td><span className="projected-club"><ClubMark name={row.team} small/><b>{row.team}</b></span><small className="position-range">Avg. finish {number(row.average_finishing_position, 1)} · 10th–90th: {number(row.position_p10, 0)}–{number(row.position_p90, 0)}</small></td><td>38</td><td>{number(row.expected_wins, 1)}</td><td>{number(row.expected_draws, 1)}</td><td>{number(row.expected_losses, 1)}</td><td>{number(row.expected_gf, 1)}</td><td>{number(row.expected_ga, 1)}</td><td>{number(row.expected_gd, 1)}</td><td className="expected-points">{number(row.expected_points, 1)}</td><td>{number(row.title_probability, 1)}%</td><td>{number(row.top4_probability, 1)}%</td><td>{number(row.relegation_probability, 1)}%</td></tr>)}</tbody></table></div>
        <div className="projected-table-legend"><span><i className="legend-top-four"/> Top four zone</span><span><i className="legend-relegation"/> Relegation zone</span><span>Avg. finish range = 10th–90th percentile</span></div>
      </section>
      <section className="surface-panel probability-distribution-panel"><div className="panel-heading"><div><span className="panel-kicker">POSITION DISTRIBUTION</span><h2>How uncertain is each club's finish?</h2><p>Average finishing position and the central 80% range across simulation runs.</p></div></div><div className="position-probability-list">{data.table.map(row => <div key={row.team} className="position-probability-row"><div><b>{row.team}</b><span>{number(row.average_finishing_position, 1)} avg.</span></div><div className="position-range-track"><i style={{left:`${(row.position_p10 - 1) / 19 * 100}%`, width:`${Math.max(2, (row.position_p90 - row.position_p10) / 19 * 100)}%`}}/></div><small>{number(row.position_p10, 0)}–{number(row.position_p90, 0)}</small></div>)}</div></section>
      <section className="surface-panel table-limitations"><b>Data and methodology notes</b><ul>{data.limitations.map((note, index) => <li key={index}>{note}</li>)}</ul><p>{data.method} Ranking uses points, goal difference and goals scored; exact residual ties are resolved deterministically rather than by a head-to-head simulation.</p></section>
    </>}
    <PageFooter />
  </div>;
}

function PageFooter() {
  return <footer className="workspace-footer"><span>© 2026 PITCHIQ · FOOTBALL INTELLIGENCE</span><span>Independent analytics project · Not affiliated with the Premier League</span></footer>;
}

function LoadingScreen() {
  return <div className="loading-screen"><span className="loading-logo"><Activity size={21}/></span><b>Connecting to PitchIQ</b><small>Reading the current football data snapshot…</small></div>;
}

function ConnectionError({ error, onRetry }) {
  return <div className="connection-error"><div className="error-symbol"><Database size={24}/></div><Eyebrow>DATA CONNECTION</Eyebrow><h2>We couldn't reach the data engine.</h2><p>{error}</p><span>Start the FastAPI backend, then retry the connection.</span><button className="button-dark" onClick={onRetry}><RefreshCw size={15}/> Retry connection</button></div>;
}

function Workspace({ overview, teams, active, setActive, onHome, onMenu, mobileOpen, onClose, onRetry, error }) {
  let content;
  if (error) content = <ConnectionError error={error} onRetry={onRetry}/>;
  else if (!overview) content = <LoadingScreen/>;
  else if (active === 'Overview') content = <Dashboard overview={overview} teams={teams} onNavigate={setActive}/>;
  else if (active === 'Match predictor') content = <PredictorPage teams={teams}/>;
  else if (active === 'Matchday predictor') content = <MatchdayPredictorPage/>;
  else if (active === 'Table predictor') content = <TablePredictorPage/>;
  else if (active === 'Team intelligence') content = <TeamIntelligencePage teams={teams}/>;
  else if (active === 'Player intelligence') content = <PlayerIntelligencePage teams={teams}/>;
  else if (active === 'Head to head') content = <HeadToHeadPage teams={teams}/>;
  else if (active === 'Model performance') content = <ModelPerformancePage overview={overview}/>;
  return <div className="workspace-shell"><Sidebar active={active} setActive={setActive} onHome={onHome} open={mobileOpen} onClose={onClose}/><div className="workspace-main"><Topbar active={active} onMenu={onMenu} onHome={onHome} overview={overview}/>{content}</div></div>;
}

function App() {
  const [mode, setMode] = useState('landing');
  const [active, setActive] = useState('Overview');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [overview, setOverview] = useState(null);
  const [teams, setTeams] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  async function loadData() {
    setLoading(true); setError('');
    try {
      const [o, t] = await Promise.all([getJSON('/api/overview'), getJSON('/api/teams')]);
      setOverview(o); setTeams(t.teams || []);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }
  useEffect(() => { loadData(); }, []);
  function enterWorkspace() { setMode('workspace'); setActive('Overview'); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  function goHome() { setMode('landing'); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  return mode === 'landing' ? <LandingPage overview={overview} teams={teams} onEnter={enterWorkspace}/> : <Workspace overview={loading ? null : overview} teams={teams} active={active} setActive={setActive} onHome={goHome} onMenu={() => setMobileOpen(true)} mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} onRetry={loadData} error={error}/>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
