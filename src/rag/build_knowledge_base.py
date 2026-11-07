import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RAG_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "rag_docs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


DOCUMENTS = [
    # --- Fault type descriptions ---
    {
        "title": "Fault Type: Pipe Burst",
        "category": "fault_types",
        "content": """
A pipe burst is one of the most severe faults in a district heating grid.
It causes a sudden and dramatic drop in supply pressure and flow rate at the
affected segment, accompanied by a sharp spike in demand as the system attempts
to compensate for heat loss. The heating demand anomaly signature is characterised
by an initial spike of 40-80% above forecast, followed by a sustained drop as
the supply side fails to keep up.

Typical causes include corrosion in older pipe segments (>20 years), ground
movement near substations, thermal expansion stress at pipe joints, and water
hammer events caused by rapid valve closures. Cold snaps accelerating from above
0°C to below -10°C within 24 hours are a common trigger.

Detection indicators: heating demand spike >40% above XGBoost forecast, sustained
for 1-4 hours. GNN fault localisation typically flags peripheral nodes with high
pipe length and low flow velocity as most likely fault locations.

Operator response procedure:
1. Immediately isolate the affected pipe segment using the nearest upstream valve.
2. Reroute flow via redundancy loop if available.
3. Dispatch field engineer to inspect flagged GNN nodes within 30 minutes.
4. Notify affected building clusters of supply interruption.
5. Initiate emergency repair protocol per Svensk Fjärrvärme standard SR 2021:04.
Recovery time estimate: 4-8 hours for temporary repair, 24-72 hours for permanent.
""",
    },
    {
        "title": "Fault Type: Supply Failure",
        "category": "fault_types",
        "content": """
A supply failure occurs when the central heat plant or a major substation
experiences a complete or partial loss of output. The anomaly signature is a
sudden drop in heating demand to near-zero levels for multiple hours, despite
cold ambient temperatures that would normally drive high demand.

This fault type is most dangerous in winter when outdoor temperatures are below
-10°C, as building temperatures can drop dangerously within hours.

Typical causes include boiler trips at the central plant, pump failures,
control system faults, and fuel supply interruptions. Primary and backup pumps
should be checked first.

Detection indicators: heating demand drop to <20% of forecast for 2+ consecutive
hours during periods when temperature is below 5°C. GNN will typically flag the
plant node (node 0) or major substation nodes as high fault probability.

Operator response procedure:
1. Check central plant status immediately via SCADA.
2. Activate backup boiler if primary is offline.
3. Check pump status — P1 primary, P2 backup.
4. If plant is operational, check for blocked valves at major substations.
5. Notify grid control centre and escalate to plant operations manager.
6. Issue customer communication if interruption exceeds 30 minutes.
Recovery time estimate: 15 minutes (pump restart) to 4 hours (boiler restart).
""",
    },
    {
        "title": "Fault Type: Insulation Degradation",
        "category": "fault_types",
        "content": """
Insulation degradation is a slow, progressive fault where pipe insulation loses
its thermal resistance over time due to moisture ingress, mechanical damage, or
material aging. Unlike acute faults, this manifests as a gradual upward drift
in heating demand over days to weeks, as heat losses from pipes increase.

The anomaly signature is a slow drift of 10-20% above forecast that persists
for 12-48 hours or longer. The XGBoost forecast will initially track this drift
using lag features, but the conformal prediction bands will eventually flag the
sustained deviation.

Typical affected segments: buried pipes in areas with high groundwater levels,
pipes near road crossings subject to traffic vibration, and segments older than
25 years. Umeå and Kiruna grids are particularly susceptible due to freeze-thaw
cycling.

Detection indicators: heating demand drift >10% above forecast, sustained for
12+ hours. Rolling mean deviation increasing monotonically over multiple days.
GNN nodes with longest pipe segments and oldest installation dates have highest
fault probability.

Operator response procedure:
1. Order ground temperature survey along suspected pipe route.
2. Check heat meter data at affected substations for anomalous return temperatures.
3. If return temperature is elevated (>60°C), insulation breach is confirmed.
4. Schedule pipe inspection using thermal imaging camera.
5. Log for planned maintenance if not critical — prioritise segments >30 years old.
Recovery time estimate: planned replacement 2-4 weeks, emergency patch 24-48h.
""",
    },
    {
        "title": "Fault Type: Stuck Valve",
        "category": "fault_types",
        "content": """
A stuck valve occurs when a control valve in the distribution network fails to
close fully, resulting in continuous low-level flow through a circuit that should
be isolated. In district cooling systems, this causes a persistent low-level
cooling demand during periods when the system should be completely offline
(typically when outdoor temperature is below 12°C).

This is one of the most common faults in district cooling networks and is
particularly important to detect in Nordic climates where cooling should be
zero for 8-9 months of the year. A stuck valve reading 0.05-0.15 MW during
winter months is a clear operational anomaly.

Typical causes include valve actuator motor failure, debris in the valve seat,
corrosion of the valve stem, and control signal faults causing the actuator to
receive incorrect position commands.

Detection indicators: cooling demand showing steady non-zero value (typically
0.05-0.15 MW) during periods when outdoor temperature is below 12°C and no
cooling should be active. The hybrid conformal detector flags this as a zero-
demand month anomaly using raw residual thresholds.

Operator response procedure:
1. Identify the stuck valve location using the GNN node fault probabilities.
2. Dispatch field engineer to inspect the valve actuator at the flagged substation.
3. Attempt remote valve closure via SCADA first.
4. If remote closure fails, perform manual override at the valve cabinet.
5. Log valve serial number and schedule replacement if actuator is damaged.
Recovery time estimate: 30 minutes (remote reset) to 4 hours (manual replacement).
""",
    },
    {
        "title": "Fault Type: Sensor Drift",
        "category": "fault_types",
        "content": """
Sensor drift is a calibration fault where a heat meter or flow sensor gradually
deviates from its true reading over time. This manifests as a slow linear
accumulation of noise in the demand signal, causing measured values to diverge
from physical reality even when the grid is operating normally.

In district heating, sensor drift is most dangerous in winter because it can
mask genuine supply failures or cause false alarms that undermine operator
trust in the monitoring system. In district cooling, sensor drift during
off-season months creates a false background signal that resembles stuck valve
faults.

Typical causes include moisture ingress into sensor housings, electromagnetic
interference from nearby electrical equipment, battery depletion in wireless
sensors, and physical damage from ground movement or maintenance activities.

Detection indicators: demand signal shows slow linear ramp during periods of
stable weather, increasing by 0.1-0.3% per hour over 1-2 week windows.
The rolling standard deviation of residuals increases monotonically. The conformal
detector flags this as sustained deviation rather than an acute event.

Operator response procedure:
1. Cross-check the suspect meter reading against adjacent meter readings.
2. Compare heat meter output against independent flow measurement if available.
3. Inspect sensor housing for moisture or physical damage.
4. Perform meter re-calibration using reference standard per EN 1434.
5. If calibration fails, replace meter and update meter data in SCADA.
Recovery time estimate: 1 hour (recalibration) to 4 hours (meter replacement).
""",
    },
    {
        "title": "Fault Type: Anti-Cyclical Demand Pulse",
        "category": "fault_types",
        "content": """
An anti-cyclical demand pulse is an anomalous brief spike in cooling demand
during periods of freezing outdoor temperatures, when mechanical cooling is
physically impossible under normal operating conditions. This fault type
indicates either a serious operational error, equipment malfunction, or
telemetry system fault.

The most common cause is a control system error where cooling pumps are
inadvertently activated during the winter shutdown period. Other causes
include data logging errors that associate a summer demand reading with a
winter timestamp, and short-circuit faults in pump control wiring.

Detection indicators: cooling demand spike of 0.3-0.8 MW during hours when
outdoor temperature is below -5°C. Duration is typically 1-3 hours. The
hybrid conformal detector is specifically calibrated to flag these events
using zero-demand month raw residual thresholds.

Operator response procedure:
1. Immediately check cooling pump status via SCADA.
2. If pumps are running, perform emergency shutdown of cooling circuit.
3. Inspect control panel for wiring faults or relay malfunctions.
4. Review SCADA event log for unauthorised control commands.
5. File incident report and notify operations manager within 1 hour.
Recovery time estimate: 15-30 minutes (pump shutdown), 2-4 hours (root cause).
""",
    },
    {
        "title": "Fault Type: Heat Exchanger Fouling",
        "category": "fault_types",
        "content": """
Heat exchanger fouling is the most common silent efficiency fault in Swedish
district heating substations. Mineral scaling (calcium carbonate, iron oxide)
gradually accumulates on the internal plate surfaces of the plate heat exchanger
(PHE), progressively reducing the rate of thermal energy transfer from the
primary district heating loop to the secondary building loop.

Unlike pipe bursts or supply failures, this fault presents as a demand DROP
rather than a spike. The heating demand falls below forecast during cold snaps
because the fouled PHE can no longer transfer sufficient thermal energy despite
normal supply pressure and temperature. This counterintuitive signature — less
heat being drawn when more is needed — is frequently missed by naive threshold
detectors that treat demand drops as safe events.

The critical dual signature: demand below forecast combined with elevated return
temperature above 60°C. Normal operation returns water at 40-55°C after heat
extraction. Fouled plates fail to extract heat efficiently, returning hot water
to the primary loop and degrading overall grid efficiency.

Secondary loop degradation: building occupants experience insufficient heating
despite the district heating system operating normally. This is often the first
complaint that triggers investigation.

Detection indicators:
- Heating demand 15-35% below forecast during cold snap (temperature <-5°C)
- Return temperature consistently above 58°C at the affected substation
- Progressive worsening over days to weeks rather than acute onset
- GNN flags the specific substation node — neighbouring nodes are unaffected

Operator response procedure:
1. Check return temperature at all substations — identify which node shows >58°C.
2. Cross-reference with GNN fault localisation output to confirm node identity.
3. Verify building loop pressure and flow are within normal operating ranges.
4. Schedule PHE chemical cleaning (citric acid flush per Alfa Laval protocol).
5. If cleaning is not sufficient, schedule PHE plate replacement.
6. Document fouling progression rate for predictive maintenance scheduling.

Recovery time: chemical cleaning 4-8 hours, plate replacement 1-2 days.
PHE cleaning should be performed annually in high-hardness water areas.
""",
    },
    {
        "title": "Fault Type: Data Centre Thermal Dump",
        "category": "fault_types",
        "content": """
A data centre thermal dump occurs when a hyperscale or HPC data centre
experiences a cooling system failure and diverts its massive waste heat stream
directly into the district heating return loop to protect server hardware from
thermal damage. This is a real operational scenario in modern green-grid
infrastructure where data centres are thermally integrated with district heating
networks (as practised by atNorth, Microsoft, and other Nordic data centre
operators).

Under normal operation, data centres extract thermal energy from the district
heating loop for server cooling and return moderately warm water. During a
thermal dump event, the data centre emergency-releases high-temperature waste
heat (70-90°C) directly into the return loop, flooding the heating network with
free thermal energy.

The unique dual-grid signature distinguishes this fault from all others:
- Simultaneous COOLING demand collapse: data centre cooling system goes offline
- Simultaneous HEATING demand collapse: the city no longer needs to generate
  heat because the data centre is flooding the return loop with thermal energy
- Return line temperature spikes above 65°C network-wide
- This appears as correlated anomalies across BOTH heating and cooling targets

This event violates return line temperature limits (maximum 60°C per Svensk
Fjärrvärme SR 2018:02) and can cause damage to distribution pumps and
residential heat exchangers not designed for high return temperatures.

HPC thermal dumping protocols require immediate coordination between the data
centre operations team and the district heating control centre.

Detection indicators:
- Simultaneous negative deviation in both heating AND cooling demand
- Return temperature spike above 65°C
- GNN localises fault to the industrial/data centre cluster node
- Anomaly onset is acute (within 1-2 hours) and correlated across grids

Industrial waste heat integration boundaries per SR 2018:02:
- Maximum permitted return temperature: 60°C
- Maximum permitted thermal dump rate: 20 MW per hour
- Mandatory notification to grid operator before any emergency dump event

Operator response procedure:
1. Confirm simultaneous anomaly in both heating and cooling targets for same city.
2. Check return line temperature — if above 65°C, activate thermal dump protocol.
3. Contact data centre operations emergency line immediately.
4. Partially close primary supply valves to reduce loop exposure to high return temp.
5. Monitor downstream substation PHE temperatures for stress indicators.
6. Log event and notify Svenska Kraftnät within 2 hours per regulatory requirement.
Recovery time: 1-4 hours (data centre cooling restart), 2-6 hours (loop stabilisation).
""",
    },
    {
        "title": "Fault Type: Inverse Diurnal Sensor Drift",
        "category": "fault_types",
        "content": """
Inverse diurnal sensor drift is an advanced telemetry calibration failure where
a degrading heat meter at a major pump substation gradually uncouples from
physical reality over a 7-14 day window. Unlike simple linear drift, this fault
produces demand readings that are phase-inverted relative to the expected
diurnal consumption pattern.

The defining signature is a temporal contradiction: demand rises during periods
of minimal consumer activity (02:00-05:00 when the city is asleep and
temperatures are stable) and drops during peak morning demand (07:00-09:00
when residents are showering and heating systems are ramping up). This is
physically impossible under normal grid operation, as consumer behaviour and
ambient temperature both drive demand in the same direction throughout the day.

The progressive nature of this fault is critical: it typically begins as a
small phase offset (1-2 hours) that worsens over two weeks until the meter is
completely phase-inverted. Early detection requires monitoring the cross-
correlation between demand patterns and expected diurnal profiles.

Diagnosis key: the temporal contradiction is the diagnostic signal. A demand
expansion during stable ambient temperatures and minimal consumer interaction
hours indicates telemetry calibration drift rather than a physical pipeline
event. No physical fault can cause demand to peak at 02:00 in a residential
district heating grid.

Remote sensor telemetry ping and zero-point calibration re-indexing should
be performed before dispatching a physical maintenance crew, as 60-70% of
inverse diurnal cases are resolved by remote recalibration.

Detection indicators:
- Demand peaks at 02:00-05:00 rather than 07:00-09:00 and 17:00-20:00
- Progressive phase shift worsening over 7-14 days
- Deviation from forecast is moderate (10-25%) but temporally structured
- Adjacent meter readings do not show same pattern (isolates the fault node)
- No correlated weather event explains the unusual demand timing

Operator response procedure:
1. Run remote telemetry ping on the suspect meter — check signal integrity.
2. Perform zero-point calibration re-index via SCADA remote command.
3. Compare pre and post recalibration readings over a 2-hour window.
4. If remote recalibration resolves the inversion, log and monitor for recurrence.
5. If fault persists, dispatch field crew for physical meter inspection and
   replacement per EN 1434 calibration protocol.
6. Flag meter serial number for accelerated replacement schedule.
Recovery time: 30 minutes (remote recalibration) to 4 hours (physical replacement).
""",
    },
    {
        "title": "Fault Type: Spring Thaw Pipe Stress",
        "category": "fault_types",
        "content": """
Spring thaw pipe stress is a Nordic-specific fault pattern that occurs during
the April-May thaw period when frozen ground rapidly expands and contracts
around buried pipe segments. This creates mechanical stress at pipe joints,
couplings, and bends that can cause micro-leaks and, in severe cases, full
ruptures across multiple adjacent pipe segments simultaneously.

Unlike single-point pipe bursts, spring thaw events produce a distributed
micro-leak pattern: multiple adjacent nodes show simultaneous small pressure
drops and modest demand increases (5-20% above forecast each), rather than
a single catastrophic spike at one location. This distributed signature is
characteristic and distinct from other fault types.

The GNN is uniquely positioned to detect spring thaw events because the
affected nodes form a spatial cluster in the graph topology — they are
geographically adjacent and connected in the network. Random isolated node
failures do not produce this clustered spatial pattern.

High-risk conditions:
- Outdoor temperature crossing 0°C after extended period below -10°C
- Rapid thaw rate: temperature rise >15°C within 48 hours
- High groundwater areas (near rivers, coastal zones)
- Pipe segments >20 years old in freeze-thaw prone soil types

Cities at highest spring thaw risk in SY1 network:
- Umeå: river proximity at nodes 5-7
- Kiruna: all segments due to extreme freeze-thaw cycles
- Gothenburg: harbour-adjacent segments

Detection indicators:
- Multiple adjacent nodes (3+) showing 5-20% demand increase simultaneously
- Outdoor temperature crossing 0°C from below during April-May
- GNN flags a spatial cluster of nodes rather than isolated nodes
- Pattern onset coincides with thaw period

Operator response procedure:
1. Confirm spring thaw conditions: check outdoor temperature trend and forecast.
2. Identify spatial cluster of flagged GNN nodes — map against pipe network.
3. Dispatch inspection team to cover all nodes in the cluster, not just the highest.
4. Check ground moisture and visible ground movement along pipe routes.
5. Increase system pressure monitoring frequency during thaw period.
6. Pre-position repair clamps and materials for affected pipe diameter range.
Recovery time: temporary clamp repair 2-4 hours per segment, permanent 1-3 days.
""",
    },
    {
        "title": "Fault Type: Control System Demand Response Misfire",
        "category": "fault_types",
        "content": """
A demand response misfire occurs when the district heating control system
sends an incorrect load reduction command to a substation cluster at the wrong
time — typically activating demand reduction during a peak heating event rather
than during off-peak hours as intended. This is a control system fault, not a
physical grid fault, and requires a fundamentally different response.

The defining signature is unnatural synchronisation: a perfectly coordinated
simultaneous demand drop across 3-5 nodes that is physically impossible from
any real pipe or equipment failure. Real physical faults propagate through the
network with staggered timing as pressure waves travel at finite speed (~1000 m/s
in pressurised water pipes). Synchronised drops across geographically separated
nodes within the same 1-minute SCADA scan window indicate a control signal event.

This fault is particularly dangerous during cold snaps because the control
system may be suppressing heating supply precisely when maximum output is needed,
causing building temperatures to fall dangerously in a coordinated pattern
across an entire district.

Detection indicators:
- Simultaneous demand drop across 3+ non-adjacent nodes (within same SCADA scan)
- Drop magnitude is proportionally identical across all affected nodes (unnatural)
- No correlated physical event (no pressure transient, no weather event)
- SCADA event log shows automated control commands preceding the demand drop
- GNN fault localisation spreads evenly across multiple nodes rather than
  clustering at a single fault origin (characteristic of control system events)

Operator response procedure:
1. Check SCADA event log immediately — look for automated demand response commands
   in the 5 minutes preceding the anomaly.
2. If demand response command found: issue manual override to cancel the command.
3. Restore normal setpoints for all affected substations via SCADA.
4. Verify building supply temperatures are recovering across affected nodes.
5. Lock out the demand response automation system pending investigation.
6. Notify control system vendor and file incident report with cause code DR-MISFIRE.
Recovery time: 5-15 minutes (manual override), 1-2 days (root cause investigation).
""",
    },
    # --- City-specific operational notes ---
    {
        "title": "City Operations: Stockholm Grid",
        "category": "city_operations",
        "content": """
Stockholm district heating grid is the largest in the SY1 system with 30
monitored nodes and 32 pipe segments covering central and northern districts.
Peak heating demand reaches 800 MW during cold snaps below -15°C.

Key operational characteristics:
- Central heat plant at node 0 supplies all 29 downstream substations.
- Three redundancy loops provide backup routing if primary segments fail.
- Oldest pipe segments (nodes 12-18) date from 1985 and require priority
  monitoring for insulation degradation.
- The Stockholm grid shows the highest forecast uncertainty due to its large
  scale — MAE of 15.3 MW means anomaly thresholds are wider than smaller cities.
- Stockholm hosts the largest data centre cluster in the SY1 network — thermal
  dump events are a documented operational risk requiring specific protocols.

Known vulnerabilities:
- Node 7 substation has a history of valve actuator faults (2 incidents in 2023).
- Pipe segment between nodes 3 and 11 crosses a major road and is susceptible
  to ground movement during spring thaw.
- Cooling demand is highest in Stockholm (peak 120 MW) due to data centre load.
- Return line temperature monitoring is critical during data centre high-load periods.

Escalation contacts: Stockholm Grid Control +46 8 XXX XXXX (24/7).
Data Centre Emergency Line: +46 8 XXX YYYY (24/7 — thermal dump events only).
""",
    },
    {
        "title": "City Operations: Kiruna Grid",
        "category": "city_operations",
        "content": """
Kiruna district heating grid operates in the most extreme climate conditions
in the SY1 system. Outdoor temperatures regularly reach -30°C to -40°C in
winter, placing maximum stress on pipe insulation and valve actuators.

Key operational characteristics:
- Smallest grid with 12 nodes and 12 pipe segments.
- Peak heating demand 100 MW — modest absolute scale but critical given climate.
- Near-zero cooling demand (1.4% of hours) means all non-zero cooling readings
  in winter should be treated as potential fault signatures.
- GraphSAGE model for Kiruna has higher uncertainty (val_loss 0.38) due to
  sparse anomaly signal — fault localisation results should be treated as
  indicative rather than definitive.
- Spring thaw risk is extreme — the most severe freeze-thaw cycling in the network.

Known vulnerabilities:
- All pipe segments are susceptible to freeze-thaw cycling — inspect annually.
- Valve actuators require heated enclosures below -25°C to prevent hydraulic
  fluid freezing.
- Supply failure risk is elevated during Polar Night (November-January) when
  solar thermal backup is unavailable.
- Spring thaw period (April-May) requires daily pressure monitoring across all nodes.

Escalation contacts: Kiruna Grid Control +46 980 XXX XXX (24/7).
""",
    },
    {
        "title": "City Operations: Gothenburg Grid",
        "category": "city_operations",
        "content": """
Gothenburg district heating grid serves 25 nodes across western Sweden's largest
city. Maritime climate means milder winters than inland cities but higher wind
speeds that amplify heat loss from buildings and pipes.

Key operational characteristics:
- Peak heating demand 450 MW driven by combination of temperature and wind chill.
- Cooling is active approximately 19% of hours — highest summer cooling load
  among mid-size cities.
- Five active cooling months (May-September) with peak cooling demand 80 MW.
- Gothenburg cooling detection has highest recall (80%) — well-calibrated.
- Harbour-adjacent pipe segments (nodes 18-22) have elevated spring thaw risk
  due to coastal groundwater and tidal influence on soil stability.

Known vulnerabilities:
- Coastal humidity accelerates pipe corrosion — higher pipe burst risk.
- Storm surge events can cause ground movement near harbour-adjacent pipe segments.
- Wind-driven rain can penetrate older valve cabinets causing actuator faults.
- PHE fouling risk is elevated due to high mineral content in local water supply.
  Annual chemical cleaning recommended for all substations.

Escalation contacts: Gothenburg Grid Control +46 31 XXX XXXX (24/7).
""",
    },
    {
        "title": "City Operations: Umeå Grid",
        "category": "city_operations",
        "content": """
Umeå district heating grid operates in a subarctic climate with long cold winters
and brief summers. The grid has 16 nodes and shows excellent anomaly detection
performance — cooling anomaly precision of 92% is the highest in the system.

Key operational characteristics:
- Peak heating demand 120 MW.
- Only 1 active cooling month (effectively July only) — cooling system is
  dormant for 11 months of the year.
- The very high cooling precision (0.92) means Umeå cooling alerts are
  highly reliable — operators should respond promptly.
- River proximity near nodes 5-7 means spring thaw creates high groundwater
  pressure against buried pipe segments — peak risk period April-May.

Known vulnerabilities:
- Node 8 is the most peripheral substation with longest pipe run — highest
  insulation degradation and spring thaw risk.
- River proximity near nodes 5-7 means groundwater levels vary seasonally.
- Freeze-thaw cycling is severe — spring thaw in April-May is highest risk.

Escalation contacts: Umeå Grid Control +46 90 XXX XXXX (24/7).
""",
    },
    # --- Standard operating procedures ---
    {
        "title": "SOP: Anomaly Response Protocol",
        "category": "procedures",
        "content": """
Standard Operating Procedure: Anomaly Response Protocol
Version: 3.2 | Effective: 2024-01-01 | Owner: Grid Operations

SCOPE: Applies to all anomaly alerts generated by the SY1 Nordic Energy Ops
Agent system for heating and cooling demand deviations.

SEVERITY CLASSIFICATION:
- Level 1 (Low): Single city anomaly, F1 score < 0.3, no GNN node confirmation.
  Action: Monitor for 2 hours, log in incident tracker.
- Level 2 (Medium): Single city anomaly confirmed by GNN (fault prob > 0.3).
  Action: Dispatch field inspection within 2 hours, notify shift supervisor.
- Level 3 (High): Multi-city anomaly or single city with GNN prob > 0.7.
  Action: Immediate field dispatch, notify operations manager, activate
  backup systems.
- Level 4 (Critical): Simultaneous anomaly in both heating AND cooling for same
  city, or return temperature spike above 65°C. Possible data centre thermal dump.
  Action: Immediate escalation to operations director, contact data centre
  emergency line, activate thermal dump protocol.

RESPONSE STEPS:
1. Acknowledge alert in dashboard within 5 minutes.
2. Review anomaly details: city, target (heating/cooling), deviation magnitude.
3. Check for simultaneous heating AND cooling anomaly — if both, escalate to Level 4.
4. Check GNN fault localisation — identify top 3 candidate nodes.
5. Query RAG knowledge base for fault type context and response procedure.
6. Classify severity per above table.
7. Execute response per severity level.
8. Document outcome in incident log.

ESCALATION:
If anomaly persists >4 hours despite Level 2 response: escalate to Level 3.
If grid stability is at risk: contact Swedish grid authority (Svenska Kraftnät).
Thermal dump events: notify Svenska Kraftnät within 2 hours per regulation.
""",
    },
    {
        "title": "SOP: Cooling System Winter Shutdown",
        "category": "procedures",
        "content": """
Standard Operating Procedure: Cooling System Winter Shutdown
Version: 2.1 | Effective: 2023-10-01 | Owner: Cooling Operations

SCOPE: Applies to all district cooling systems in the SY1 network.
Shutdown is triggered when outdoor temperature forecast is below 10°C for
5+ consecutive days (typically October-November).

SHUTDOWN PROCEDURE:
1. Verify all cooling demand has dropped below 0.5 MW for 48 consecutive hours.
2. Close primary cooling supply valve at each substation — verify via SCADA.
3. Drain and blow-out exposed pipe sections in Kiruna and Umeå grids (frost risk).
4. Set all cooling valve actuators to winter position (fully closed + locked).
5. Disable cooling pump circuits — lock out per LOTO procedure.
6. Update SY1 system configuration: set zero-demand month flags for affected cities.
7. Perform final inspection of valve cabinet seals before winter.

MONITORING DURING WINTER SHUTDOWN:
The SY1 hybrid conformal detector continues monitoring for:
- Stuck valve faults (steady 0.05-0.15 MW during shutdown)
- Sensor drift (slow linear ramp in meter readings)
- Anti-cyclical pulses (brief spikes during freezing conditions)

Any non-zero cooling reading during shutdown should be treated as a fault
until proven otherwise. Response per Fault Type: Stuck Valve procedure.

RESTART PROCEDURE:
Spring restart (typically April-May) — reverse the above in order 5 to 1.
""",
    },
    {
        "title": "SOP: GraphSAGE Fault Localisation Interpretation",
        "category": "procedures",
        "content": """
Standard Operating Procedure: Interpreting GNN Fault Localisation Results
Version: 1.0 | Effective: 2024-06-01 | Owner: AI Operations

SCOPE: Guidance for operators interpreting GraphSAGE node fault probability
scores generated by the SY1 anomaly detection system.

INTERPRETING FAULT PROBABILITIES:
The GraphSAGE model outputs a fault probability (0.0-1.0) for each grid node.
These probabilities indicate the likelihood that a detected anomaly originated
at or near that node, based on network topology and node characteristics.

Probability ranges:
- 0.0-0.1: Low — background noise, unlikely to be fault origin
- 0.1-0.3: Moderate — worth inspecting if field resources are available
- 0.3-0.6: High — prioritise for inspection within 2 hours
- 0.6-1.0: Critical — immediate inspection required

SPATIAL CLUSTERING INTERPRETATION:
When GNN flags a geographic cluster of adjacent nodes simultaneously, this
indicates a distributed fault pattern rather than a single-point failure.
Possible causes:
- Spring thaw pipe stress (April-May, multiple adjacent segments affected)
- Control system demand response misfire (synchronised across non-adjacent nodes)
- Data centre thermal dump (affects nodes connected to industrial cluster)

IMPORTANT LIMITATIONS:
- GNN models for Kiruna (val_loss 0.38) have higher uncertainty than other cities.
- The GNN was trained on synthetic fault patterns — novel faults may produce
  unexpected localisation results.
- Always cross-check GNN results with SCADA live data before dispatching crew.
- Node 0 (central plant) appearing in top-3 suggests supply-side fault.
- Evenly spread probabilities across many nodes suggests control system event.

DOCUMENTATION:
After each field inspection, record actual fault location and compare with GNN
prediction to support ongoing model improvement.
""",
    },
    {
        "title": "SOP: Data Centre Thermal Dump Protocol",
        "category": "procedures",
        "content": """
Standard Operating Procedure: Data Centre Emergency Thermal Dump Response
Version: 1.0 | Effective: 2024-03-01 | Owner: Industrial Integration Team

SCOPE: Applies when a thermally integrated data centre initiates an emergency
waste heat dump into the district heating return loop. Relevant for Stockholm
and any future SY1 cities with HPC or hyperscale data centre integration.

TRIGGERING CONDITIONS (any one sufficient):
- Simultaneous negative deviation in both heating AND cooling demand for same city
- Return line temperature rising above 63°C (warning) or 65°C (critical)
- Data centre emergency notification received via hotline
- SY1 agent reports Level 4 severity event

IMMEDIATE RESPONSE (first 15 minutes):
1. Confirm dual-grid anomaly: check both heating and cooling dashboards for city.
2. Check return line temperature sensor at primary distribution point.
3. Call data centre emergency operations line — confirm thermal dump is in progress.
4. Activate partial supply valve throttling: reduce primary supply by 20% to
   limit hot return water exposure to sensitive equipment.
5. Alert all downstream residential substations: PHE stress risk elevated.

STABILISATION (15-60 minutes):
6. Monitor return temperature continuously — target return to below 60°C.
7. Coordinate with data centre on thermal dump rate reduction schedule.
8. Check PHE outlet temperatures at all substations for over-temperature events.
9. If return temperature exceeds 68°C: activate emergency bypass to bleed
   hot return water to thermal buffer storage.

RETURN LINE TEMPERATURE LIMITS (per SR 2018:02):
- Normal operation: 40-55°C
- Warning threshold: 60°C — increase monitoring frequency
- Critical threshold: 65°C — activate thermal dump protocol
- Emergency threshold: 68°C — activate bypass to thermal storage

REGULATORY NOTIFICATION:
Notify Svenska Kraftnät grid operations within 2 hours of any Level 4 event.
File full incident report within 24 hours including thermal dump volume (MWh).
""",
    },
    {
        "title": "SOP: Spring Thaw Inspection Protocol",
        "category": "procedures",
        "content": """
Standard Operating Procedure: Spring Thaw Pipe Inspection Protocol
Version: 2.0 | Effective: 2024-02-01 | Owner: Asset Integrity Team

SCOPE: Applies during spring thaw period (typically 15 March - 31 May) when
freeze-thaw cycling creates elevated pipe stress risk across all SY1 grids.
Highest risk cities: Kiruna, Umeå, Gothenburg (harbour segments).

ACTIVATION TRIGGERS:
- Outdoor temperature rising above 0°C after period below -10°C
- Temperature rise rate exceeding 10°C in 24 hours
- SY1 agent detecting spatial cluster of adjacent node anomalies
- Groundwater level alert from municipal monitoring system

PRE-THAW PREPARATION (when forecast shows imminent thaw):
1. Review pipe age map — flag all segments >20 years old for priority inspection.
2. Pre-position repair clamps for pipe diameters 100mm, 150mm, 200mm.
3. Brief field teams on spring thaw signature: multiple simultaneous small
   pressure drops across adjacent nodes rather than single burst event.
4. Increase SCADA scan frequency from 15 minutes to 5 minutes.

DURING THAW INSPECTION:
5. Daily walk inspection of high-risk segments in Kiruna and Umeå.
6. Thermal imaging survey of buried pipe routes in affected areas.
7. Monitor SY1 GNN output for spatial node clustering patterns.
8. If 3+ adjacent nodes show simultaneous pressure anomalies: activate
   spring thaw response — do not treat as isolated events.

POST-THAW REVIEW:
9. Document all segments showing anomalous behaviour during thaw.
10. Update pipe replacement priority list based on thaw season performance.
11. File annual spring thaw report with Energimarknadsinspektionen.
""",
    },
    # --- Equipment reference ---
    {
        "title": "Equipment: Heat Exchangers and Substations",
        "category": "equipment",
        "content": """
District heating substations in the SY1 network contain plate heat exchangers
(PHE) that transfer thermal energy from the primary district heating loop to
the secondary building loop. Substation capacity ranges from 0.5 MW to 5.0 MW
per unit.

COMMON SUBSTATION FAULTS:
1. PHE fouling: scale buildup reduces heat transfer efficiency. Signature:
   demand below forecast + return temperature above 58°C. Most common fault.
   Diagnosis key: demand DROP during cold snap = fouling, not a safe event.

2. Control valve failure: see Fault Type: Stuck Valve documentation.

3. Circulation pump failure: building-side pump stops, causing no heat
   delivery despite normal supply pressure. Signature: sudden demand drop
   at a single node without affecting neighbouring nodes.

4. Temperature sensor failure: incorrect readings causing control system
   to over- or under-supply heat. Signature: demand spike or drop without
   correlated weather event. Check for inverse diurnal pattern.

PHE FOULING DIAGNOSIS:
Return temperature is the key diagnostic. Normal return: 40-55°C.
Return temperature above 58°C indicates fouling. Above 60°C is critical.
Cross-check: if demand is low but return temperature is high, PHE fouling
is confirmed. Do not dismiss demand drop as a positive event.

MAINTENANCE SCHEDULE (Svensk Fjärrvärme SR 2019:01):
- Annual: visual inspection, valve actuator function check, return temp review.
- Biennial: PHE chemical cleaning (citric acid flush per Alfa Laval protocol).
- 5-year: full PHE plate replacement if efficiency has dropped >15%.
- 10-year: complete substation audit including pipe connections and insulation.

PERFORMANCE BENCHMARKS:
- Minimum PHE efficiency: 85% heat transfer at rated capacity.
- Maximum return temperature: 55°C at design outdoor temperature.
- Maximum pressure drop across PHE: 0.3 bar at rated flow.
""",
    },
    {
        "title": "Equipment: Pipe Network Specifications",
        "category": "equipment",
        "content": """
The SY1 district heating pipe network uses pre-insulated steel pipes
(bonded system per EN 253) for all primary distribution. Pipe diameters
range from 100mm to 400mm depending on flow capacity requirements.

PIPE SPECIFICATIONS:
- Supply temperature: 80-110°C depending on outdoor temperature
- Return temperature: 40-60°C (lower is better — efficient heat extraction)
- Design pressure: 16 bar maximum working pressure
- Heat loss standard: 15-30 W/m for properly insulated pipes

PIPE AGE AND RISK:
- <10 years: low risk, normal monitoring
- 10-20 years: medium risk, annual inspection recommended
- 20-30 years: high risk, biennial inspection, thermal imaging survey
- >30 years: critical risk, replacement planning required

SPRING THAW PIPE STRESS:
Pre-insulated bonded pipe systems are particularly susceptible to spring thaw
mechanical stress at coupling joints. The foam insulation layer can trap
moisture during freeze cycles, creating ice lenses that expand at thaw.
Risk is highest for pipes installed before 2000 that predate current
EN 253 moisture barrier standards.

FLOW VELOCITY STANDARDS:
- Normal operation: 0.5-2.0 m/s
- Below 0.5 m/s: risk of sedimentation and biological growth
- Above 2.0 m/s: elevated erosion risk, noise generation

EMERGENCY PIPE REPAIR:
For pipe bursts requiring emergency repair outside business hours, contact
the on-call pipe repair contractor listed in the emergency contact register.
Temporary clamp repair is acceptable for pipes <200mm diameter.
For pipes >200mm, full section replacement is required per EN 253.
""",
    },
    {
        "title": "Equipment: Return Line Temperature Management",
        "category": "equipment",
        "content": """
Return line temperature management is critical for district heating grid
efficiency and equipment protection. The return temperature measures how
effectively thermal energy has been extracted from the primary loop before
water is returned to the heat plant for reheating.

RETURN TEMPERATURE BENCHMARKS:
- Optimal: 40-45°C (maximum heat extraction, minimum reheating cost)
- Normal: 45-55°C (acceptable operational range)
- Warning: 55-60°C (investigate — possible PHE fouling or overcirculation)
- Critical: 60-65°C (action required — PHE cleaning or thermal dump event)
- Emergency: >65°C (immediate response — data centre thermal dump suspected)

CAUSES OF ELEVATED RETURN TEMPERATURE:
1. PHE fouling (most common): scaling prevents efficient heat extraction.
   Solution: chemical cleaning or plate replacement.
2. Overcirculation: secondary loop flow rate too high, insufficient dwell time
   for heat extraction. Solution: reduce secondary pump speed.
3. Data centre thermal dump: emergency waste heat injection into return loop.
   Solution: activate thermal dump protocol, contact data centre emergency line.
4. Mixing fault: hot supply water bypassing the PHE and entering return loop.
   Solution: check bypass valves and mixing valve positions.

MONITORING REQUIREMENTS:
- Return temperature must be logged at 15-minute intervals at all substations.
- Automated alert when return temperature exceeds 58°C at any substation.
- Daily review of return temperature trend for all nodes — rising trend over
  5+ days indicates developing PHE fouling requiring scheduled maintenance.

EFFICIENCY METRIC:
Delta-T (supply temperature minus return temperature) is the primary
efficiency indicator. Target delta-T: 50-65°C. Delta-T below 30°C indicates
a serious heat extraction problem requiring immediate investigation.
""",
    },
]


def chunk_document(
    doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list:
    text = doc["content"].strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "title": doc["title"],
                    "category": doc["category"],
                    "chunk": chunk,
                    "doc_id": doc["title"].replace(" ", "_").replace(":", ""),
                }
            )
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_knowledge_base():
    os.makedirs(RAG_DOCS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== Building RAG Knowledge Base ===")
    print(f"Documents: {len(DOCUMENTS)}")

    all_chunks = []
    for doc in DOCUMENTS:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['title']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["chunk"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(
        texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
    )
    embeddings = embeddings.astype(np.float32)
    print(f"Embedding shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    index_path = os.path.join(MODELS_DIR, "rag_faiss.index")
    chunks_path = os.path.join(RAG_DOCS_DIR, "chunks.json")
    model_path = os.path.join(MODELS_DIR, "rag_embedder_name.pkl")

    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    with open(model_path, "wb") as f:
        pickle.dump(EMBEDDING_MODEL, f)

    print(f"\nSaved:")
    print(f"  FAISS index:  {index_path}")
    print(f"  Chunks:       {chunks_path}")
    print(f"  Model name:   {model_path}")
    print(f"\nKnowledge base ready. {len(all_chunks)} retrievable chunks.")


if __name__ == "__main__":
    build_knowledge_base()
