// Electron plasma periods use fp = omega_pe/(2 pi), not omega_pe itself.
export function simulationClock(frame,frameDtSeconds,omegaPe){
 if(![frame,frameDtSeconds,omegaPe].every(Number.isFinite)||frame<0||frameDtSeconds<=0||omegaPe<=0)throw Error('Invalid simulation clock');
 const seconds=frame*frameDtSeconds;
 return {nanoseconds:seconds*1e9,plasmaPeriods:seconds*omegaPe/(2*Math.PI)};
}
