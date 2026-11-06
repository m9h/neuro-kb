# Questions for Openwater (to be raised with Dan Blizinski)

Context: neurotech@berkeley student device development on an open fUS/LIFU bench (see
`paper/main.tex`). We intend to replicate Openwater's phantom, tank and registration setup so our
results are comparable with theirs and stay inside the Openwater Patent Pledge.

1. **Hackathon reference meshes.** `openwater-reference-meshes` describes skull, head and brain STL
   meshes from the April 2026 Openwater Neuro Hackathon with per-mesh provenance YAMLs, but only
   `meshes/head/texturedMesh.obj` is uploaded. Are the skull and brain meshes coming? What is the
   source scan (CT? MR? which subject/dataset?) and the segmentation chain? CT-derived skulls with
   provenance under CC BY 4.0 would replace our IDC + SimNIBS route for first phantom work.
2. **Origin of the head OBJ.** Is `texturedMesh.obj` an OpenLIFU-3DScanner capture? If so, is
   there a matching skull/MR in the same frame, so it can serve as a full registration test case
   for the 3DScanner + 3DRegistration workflow?
3. **Phantom acoustic properties.** Do the gelatin/agarose and Gen2 acousto-optic phantoms in
   `openwater-phantoms` have measured speed of sound, attenuation and B/A? Is there a
   skull-mimicking phantom (3D-printed or cast) in use or planned?
4. **Rat and vessel moulds.** Flow-loop details for the rat/vein phantom (tubing, pump, flow
   rates, scatterer recipe) so we can use it for Doppler and ULM validation of the imaging bench.
5. **Verification tank hardware.** `openlifu-verification-tank` has the control software; is the
   tank/positioner CAD published (in `openlifu-mechanical`?) and which hydrophone and preamp do
   you use, so our characterisation purchases match yours?
6. **Open-LIFU hardware for a student group.** Availability and price of the console + 64-element
   array; can the console be externally triggered (TTL or over the SDK) so sonications can be
   interleaved with imaging frames from a separate acquisition system?
7. **Open-Motion.** Is sensor + console hardware available, and has it been run concurrently with
   Open-LIFU sonication?
8. **Patent Pledge.** The README summarises the pledge and lists 68 filings (table has 69 rows).
   Could we have the pledge PDF text to quote exactly, and confirmation that a CERN-OHL-S derived
   hardware design plus Apache/AGPL software qualifies?
9. **Licence intent.** docs.openwater.health says AGPL 3.0 software / CC BY-SA 4.0 hardware; the
   repos carry Apache-2.0, AGPL-3.0 and CERN-OHL-S-2.0. Which is authoritative for derivatives?
10. **Contribution.** Would Openwater accept aarch64 (DGX Spark) containers, Fedora/Debian
    packaging of the Python stack, and a Holoscan integration as upstream contributions, under the
    CLA?
