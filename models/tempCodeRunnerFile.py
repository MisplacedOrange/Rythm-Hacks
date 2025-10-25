1. Clone PlotNeuralNet if you haven't:
   git clone https://github.com/HarisIqbal88/PlotNeuralNet.git

2. Update the 'plotneuralnet_path' variable in this script to point to your PlotNeuralNet directory

3. Run this script to generate the .tex files

4. Compile the LaTeX files:
   cd network_visualization
   pdflatex heart_disease_mlp_simplified.tex
   
5. View the generated PDF!

Requirements:
- Python packages: None extra needed (uses PlotNeuralNet's modules)
- System: LaTeX installation (texlive, miktex, etc.)
""")