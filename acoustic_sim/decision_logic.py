def evaluate_beamformer_trust(coherence: float, papr_db: float, isl_db: float, tapering_applied: bool) -> str:
    """
    A programmatic decision tree for evaluating if a beamformer output is trustworthy.
    This replaces manual visual inspection of 300-point heatmaps.

    Args:
        coherence (float): Spatial coherence [0, 1] pre-beamforming.
        papr_db (float): Peak-to-Average Power Ratio of the output map in dB.
        isl_db (float): Integrated Sidelobe Level in dB.
        tapering_applied (bool): Whether spatial tapering (e.g., Hanning) was used.

    Returns:
        str: Status code ('TRUST', 'REJECT_NOISE_FLOOR', 'REJECT_SMEARED', 'WARNING_SIDELOBES')
    """

    # 1. First Gate: The Wavefield Physics Check (Microphone Level)
    # If spatial coherence is too low, the signal is destroyed by electrical noise
    # or extreme path loss before the beamformer even starts.
    if coherence < 0.10:
        return "REJECT_NOISE_FLOOR"

    # 2. Second Gate: The Sharpness Check (Post-Beamforming)
    # If PAPR is very low, the beamformer failed to focus. The energy is smeared everywhere.
    # This happens in the extreme Near-Field (wavefront mismatch) or if noise overwhelmed the array.
    if papr_db < 5.0:
        return "REJECT_SMEARED"

    # 3. Third Gate: The Sidelobe Check
    # If we have a sharp peak (High PAPR), is it the *only* peak?
    # If ISL is positive (> 0 dB), it means there is more energy in the sidelobes
    # than the mainlobe. This is severe spatial aliasing (Grating Lobes).
    if isl_db > 0.0:
        # If tapering was NOT applied, the user should retry with tapering
        # to try and push the ISL down.
        if not tapering_applied:
            return "WARNING_SIDELOBES_RETRY_WITH_TAPERING"
        else:
            return "REJECT_SEVERE_ALIASING"

    # 4. Final Trust Boundary
    # If Tapering IS applied, we expect a slightly lower PAPR threshold because
    # tapering physically widens the mainlobe.
    papr_trust_threshold = 8.0 if tapering_applied else 10.0

    if papr_db >= papr_trust_threshold:
        return "TRUST"
    else:
        return "MARGINAL_TRUST"
