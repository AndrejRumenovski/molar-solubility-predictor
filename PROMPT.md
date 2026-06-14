# **Project Specification: Cheminformatics ML Pipeline for Aqueous Solubility (LogS) Prediction**

## **1\. Project Overview & Chemical Engineering Relevance**

Aqueous solubility (![][image1], where ![][image2] is solubility in moles per liter) is a fundamental physical property of organic molecules. It dictates how compounds behave in biological systems, chemical manufacturing, and environmental transport.

Traditionally, measuring solubility requires expensive, time-consuming wet-lab experiments. In modern chemical engineering, **Quantitative Structure-Property Relationship (QSPR)** models use machine learning to predict physical properties directly from molecular structures.

This project implements an end-to-end Machine Learning pipeline that:

1. Auto-retrieves molecular structures represented as text-based **SMILES strings** (Simplified Molecular-Input Line-Entry System).  
2. Uses **RDKit** to calculate physical-chemical descriptors (molecular weight, hydrophobicity, aromaticity, rotatable bonds).  
3. Trains regression models to predict ![][image1].  
4. Provides an interactive web interface for chemical engineers to design and test new molecules in real-time.

## **2\. Technical Stack**

The application will be developed using a standard Python-based scientific stack:

* **Runtime Environment:** Python 3.10+  
* **Cheminformatics Engine:** rdkit (Open-source library for chemical informatics)  
* **Machine Learning:** scikit-learn (scikit-learn offers robust, out-of-the-box regression models)  
* **Data Engineering:** pandas and numpy  
* **Visualizations:** matplotlib, seaborn, and interactive plotly  
* **Web Dashboard:** streamlit (For rapid deployment of an interactive, shareable UI)

## **3\. Data Source Spec (Easy to Retrieve)**

We will use the **Delaney (ESOL) Dataset**, which is the gold standard benchmarking dataset for solubility. It is public, clean, and hosted reliably online.

* **Source URL:** https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv  
* **Shape:** 1,128 compounds.  
* **Key Columns:**  
  * smiles: The raw chemical string (e.g., OCC for ethanol, c1ccccc1 for benzene).  
  * measured log solubility in mols per litre: The ground truth target variable (![][image1]).  
  * ESOL predicted log solubility in mols per litre: The historical benchmark calculated by Delaney's physical equation.

## **4\. Pipeline Architecture**

The system follows a modular, object-oriented design to facilitate debugging, unit-testing, and easy integration with Streamlit:

                  ┌─────────────────────────────────────┐  
                  │      1\. DATA INGESTION              │  
                  │  Auto-downloads Delaney ESOL CSV    │  
                  └──────────────────┬──────────────────┘  
                                     │  
                                     ▼  
                  ┌─────────────────────────────────────┐  
                  │      2\. CHEMINFORMATICS FEATURIZER  │  
                  │  RDKit: Computes LogP, MW, RotB,    │  
                  │  Aromatic Proportion, PSA           │  
                  └──────────────────┬──────────────────┘  
                                     │  
                                     ▼  
                  ┌─────────────────────────────────────┐  
                  │      3\. ML TRAINING ENGINE          │  
                  │  Train/Test Split \-\> Scale Features │  
                  │  Train RF, GradBoost, & Ridge Reg   │  
                  └──────────────────┬──────────────────┘  
                                     │  
                                     ▼  
                  ┌─────────────────────────────────────┐  
                  │      4\. MODEL EVALUATION            │  
                  │  Computes R², MSE, MAE              │  
                  │  Generates parity & error plots     │  
                  └──────────────────┬──────────────────┘  
                                     │  
                                     ▼  
                  ┌─────────────────────────────────────┐  
                  │      5\. INFERENCE & STREAMLIT UI    │  
                  │  User enters custom SMILES \-\> UI    │  
                  │  draws molecule & predicts LogS     │  
                  └─────────────────────────────────────┘

## **5\. Feature Engineering (The Chemistry Logic)**

Using **RDKit**, the raw smiles string must be parsed into a molecular object, from which we extract five physical-chemical descriptors that drive solubility:

1. **Wildman-Crippen Octanol-Water Partition Coefficient (![][image3]):** A measure of hydrophobicity (how much a molecule prefers oil/solvent over water). Calculated using rdkit.Chem.Crippen.MolLogP.  
2. **Molecular Weight (![][image4]):** Larger molecules are typically harder to dissolve. Calculated using rdkit.Chem.Descriptors.MolWt.  
3. **Number of Rotatable Bonds (![][image5]):** Measures molecular flexibility. Calculated using rdkit.Chem.Lipinski.NumRotatableBonds.  
4. **Aromatic Proportion (![][image6]):** Flat, highly rigid aromatic rings pack tightly into crystals, decreasing water solubility.  
   ![][image7]  
   Calculated by counting aromatic atoms (atom.GetIsAromatic()) and dividing by the total heavy atoms (mol.GetNumHeavyAtoms()).  
5. **Polar Surface Area (![][image8]):** High polar surface area increases hydrogen bonding with water, increasing solubility. Calculated using rdkit.Chem.Descriptors.TPSA.

## **6\. Model Evaluation & Metrics**

To demonstrate engineering rigor, the pipeline must calculate and log the following validation metrics for all models on the test set (20% split):

* **Coefficient of Determination (![][image9]):** Goal target: ![][image10]  
* **Mean Squared Error (![][image11])**  
* **Mean Absolute Error (![][image12]):** Goal target: ![][image13] log units

## **7\. Interactive UI & Visualization Plan**

The Streamlit dashboard must render four specific components:

### **A. Performance Dashboard**

* **Interactive Parity Plot (Predicted vs. Experimental):** A Plotly scatter plot with a ![][image14] reference line. Hovering over a dot should display the molecule's chemical name or SMILES string.  
* **Model Comparison Metrics:** A clean table comparing ![][image9] and ![][image12] across Random Forest, Gradient Boosting, and Ridge Regression.  
* **Feature Importance Chart:** A bar chart showing which descriptor (e.g., ![][image3] vs. ![][image4]) is most influential in predicting solubility.

### **B. Live Inference Playground**

* **Input Box:** Let the user paste a custom SMILES string (e.g., Aspirin: CC(=O)Oc1ccccc1C(=O)O).  
* **Validation Check:** Clean error handling if the SMILES string is invalid.  
* **2D Molecule Renderer:** RDKit must dynamically draw and render the 2D skeletal structure of the molecule on the screen.  
* **Real-time Prediction:** Compute the descriptors on the fly and display the predicted ![][image1] with a physical explanation of what the score means (e.g., highly soluble vs. poorly soluble).

## **8\. Reproducibility & Code Standards**

* **Random Seeds:** All stochastic operations (such as train-test splitting and Random Forest initializations) must use random\_state=42 to guarantee identical runs.  
* **Error Handling:** Include robust try-except blocks around RDKit's SMILES parsing, returning None if an invalid SMILES string is inputted, ensuring the app does not crash.  
* **File Structure:** Build this as a single-file application (app.py) for easy local running, or segment it into a clean project directory. Let's aim for a self-contained, single-file streamlit implementation to maximize portability.

## **9\. Agentic Prompt Roadmap**

Copy and paste these exact prompts into your coding assistant (Cursor, Claude, etc.) one by one to build this project from scratch.

### **Phase 1: Environment & Raw Data Ingestion**

**Prompt:** \> "I want to build a Python-based machine learning pipeline that predicts molecular aqueous solubility from SMILES strings using the Delaney ESOL dataset. First, write a script to auto-download the dataset from 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv'. Inspect the data columns using Pandas and output basic dataset statistics (number of compounds, distribution of measured log solubility). Make sure to handle caching so we don't redownload the file every time the script runs."

### **Phase 2: Cheminformatics Featurization with RDKit**

**Prompt:** \> "Now, integrate RDKit to parse the SMILES strings in our dataset. Write a function that takes a SMILES string and returns a dictionary of the following 5 descriptors:

1. MolLogP (octanol-water partition coefficient)  
2. MolWt (molecular weight)  
3. NumRotatableBonds (number of rotatable bonds)  
4. AromaticProportion (Number of aromatic atoms divided by total number of heavy atoms)  
5. TPSA (Topological Polar Surface Area)  
   Handle invalid SMILES strings safely by returning None if RDKit cannot parse the string, and filter out any failed rows. Append these five new features as columns to our Pandas DataFrame."

### **Phase 3: Machine Learning Engine & Validation**

**Prompt:** \> "Write the machine learning engine. Use scikit-learn to split our featurized dataset into a 80% training set and a 20% test set, using random\_state=42. Scale the features using StandardScaler. Train three distinct models: Random Forest Regressor, Gradient Boosting Regressor, and a Ridge Regression model. Calculate the R-squared (R2), Mean Squared Error (MSE), and Mean Absolute Error (MAE) for each model on the test set. Print a clean, formatted comparison table to the console."

### **Phase 4: Streamlit UI, 2D Molecular Drawing, & Dynamic Inference**

**Prompt:** \> "Let's bundle the entire codebase into an interactive Streamlit application.

1. Set up a page layout with a sidebar containing model comparison statistics and a feature importance plot.  
2. Create a main page with a text box where users can type or paste a custom SMILES string.  
3. When a user inputs a SMILES string, use RDKit to render its 2D chemical structure in the Streamlit app using rdkit.Chem.Draw.MolToImage.  
4. If the SMILES is valid, featurize it, run it through our trained Random Forest model, and print the predicted LogS.  
5. Provide a simple text interpretation explaining if the molecule is highly soluble (LogS \> 0), moderately soluble (-4 \< LogS \<= 0), or poorly soluble (LogS \<= \-4)."

### **Phase 5: Polishing, Styling, and Plotly Visualizations**

**Prompt:** \> "Let's make the Streamlit UI look incredibly professional and polished.

1. Implement an interactive Plotly scatter plot (parity plot) comparing the predicted vs. experimental LogS values for the test set. Ensure that hovering over a point displays its SMILES string.  
2. Add custom CSS styles to make the metric cards pop.  
3. Write a brief 'How It Works' section at the bottom explaining the chemical engineering concept of QSPR, molecular descriptors, and how RDKit extracts information from chemical strings."

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAaCAYAAAAqjnX1AAADJklEQVR4Xu2WS2hTQRSGb0kFRfEdU9sk0zxAqu7ig4KLIiqK6EIFhRZcBKqU4sIHYnXjQkRwIbooiBuRoqCiUAQX4hsEBbFgEbSlFoqiUMFFC0Vq/E7uzGUyuc0q2oX54efe+c9/Z87MnJnE82r4XxGNRhdkMpkVvEbc2N+AjNfc3NyWSCS2pdPpRaI1NTUty2azUdfrJZPJ1UqpIViAg42NjctdTzWRy+XmMM4ZOMzYR3h2wTfwIkk/59niflME5iXa2Eezzo1XEfWM1QtvxWKx+UaUFSTB14z/VFbY/iCAZA/HMR5yY9UEY7TCUbZ4bUisB1529QAE2+EvZrjJjVUTjHEOjrFy8ZDYCZLf7eoBZAaqQj2mUqkYE9glRW5vk4MInlwlDzt1nXEKPE95zgGNx+NZuNTWAlj1eIdmvR2jswbR4T3eD8hs4QiJrLN9tNejv5OVor9Oni/hd745aPt0HwXNafgY5jnRC21fGdQM9cjqIav38KqcSC3X0e4jkYfMep72bUD7gdZhvqXdDX+jbTGaQPpBO68TlEQNB8JKIIAKr8d6tGvwi2yDpZstG4UrJVGe/fCDlITxqAq1pxEh3kJfF3j+lETdRSqBCqlH6QCOK6cE6Ggu2gOTpPJP65T0YTxSj7QfiU/8Wo7oyZZdb3i2K3/VT7qxImaqRzkkKmR26Gn0r/Cm56/2HvHB9hBPj6NdkW+MZsVyxCbsPkqgrHpku1bxflrrO2RwSdb24zus/NLYKm09mUk5OMYj1wjatF2PWuu3VjYAvg74kZ1MuLEidDJTUo88u+T0iS4f0P4Eu40Xzxraw/CYp7dNa9/M/SZbKgMqpx6VX6MT9L/RaMaPPgD32noJ5A8FhrfwCbxhfugFyp/AGLytKWWx2XPqSvtG4GcSvK/8/wFBPcrPHO278Dgc1O95vJfkG3z73T7LINeCJGtdMyUxObV0tNiNhcHUY9I6BHIDmBtC+pM7lvg++mwL2/6qggEa4Fl7C6VcSHISttreWQOJHFX+yc7rthzCIVapN2xXZgVyMJT/k/kCviK5Zzx3ev/oT3MNNdRQAX8AufjrajEUXmEAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAaCAYAAACHD21cAAABU0lEQVR4Xu2TO0vEUBCFI1lBURSUEMibkGbBLoLYWS1WNgoK2m0hWFgpotiKCBaihbWdjZW9iKVWFjaCoOAfsNjSxzeXROJw/4DigcPcnTNn7mSSdZzfD8/zhrMsm4njuJPn+ajkwjAcL4rC07UGZVn2p2m6C5+SJFknrsE7eEijG2JbewQtik/hue/7Q3VSbsJ0i+laJmkaDBCm4QvjTVi0HXis8wYIe/CVGyKLtknDOZ03YJwzCj6J2/x0m1oURQUca+a+gWFJjBXf4RXssskRXfsDslEWs1+Z6gbCe9v4NrgUt5nggPgmZs6rukjgyjMQ+7SAYRbjB5Nsac0hmSOecGxZtBKtB5e15siaES7pPqA1jCvwMQiCWGv1++thnGrmZXxZDJxv5g3kE0K4gBvwoTp3ueWI+EyzRcfy7NJ1sFqMeR2MPYlpQf4ZttH/8bfwBdayTX0Gkw5PAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAaCAYAAAAaAmTUAAADK0lEQVR4Xu1WO2hUQRR9SxJR/KPLmuzn7i4ri6tgsVppE1FJiiDYKARBENRCLJRoK0KwsAuCEJVgEaIoWAmCFgFBBAsttIksRomRGBQUImjQeE5mhr0Z3+pGNiHFO3B4b+65M3Pnztx5LwgiRIjw3xCRdnASnFH8AT5MJpMbfP9GA/McAz97839UtgmwJ5VKrfD71gQ6XAd/ZbPZDl9baCQSiZWY+xH4IZPJ5JUUQzwHYP8O+1W0m5UWDjiuR4dn4CgykPT1hQZOQApzj3FBXJjWYGsF34KfwC1aCwWdrPPdoJ7VNxhI5l6eCrA3RMtzx0J2LRxw7AZn4Hze12ohl8sl4N+VTqf3+9l0oJ06/MpoNvm6A+fl/GCnr+GYHbbaUFBPouHYB05j0N2+5gODb+IOgvfsRD3gGwS9Q7k1YazTsL/A8zieveBz+uEYb1Z+HG857PclJPNImMA2Au0l37UWClUvFWbb1zU4IAcG+8vlcos1x9AexDgP7I0Tw/s52MbQLtCBduqwvWpra9tYHXFWS8I+Ck7CZ4Bjk1jkbTE32eVCobBG96kJ+Ue9lEqlZTbwZjE33rgL0gET3xRTpK0IaCsDA69AilFXCRt0NodMtV6usb8jE6sSVh/E1gt4xtcCk/WLOELbpMai1TGZXQzaFxgcg3Q+CGw7bF+hnXA2B1UvB31t3hCT7dB64Q5Au2GPSRcn9QPiORdz27BAW/AcZFuffzEJ+4ak7FRddSLqu3b/BrX9f3xfuMVizm8323h2cjFclPZDQCfFJGOfbfPIPS4Wi6utC2toQELqRX1fhuPx+CqtzRuYpIyBpsQ7OmLObT/0d3jmaEMgaby/Bk+p/qyPCng2sLVgF1exyeEX/BDaPyWkXqB1iKmXPm2fF2zRjUv1X4iTMUMk3519zt0uZnfoc8eSu7onUEFyRzH+Jdjf4zkC3sL7tD6esB0R8//l5iF5VNudz6KAwfKmQXDrfC0MYupliqfA15Y0EPAu3mb4LsStidf5EPg0n8+vneO8lKH+fCewc8XA1MtRtL9II67dxQb/wxD4E3BYzK/MgF1YhAgRIjQevwFitAL+slCgIwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAAC4ElEQVR4Xu2WO2hUQRSGbzBCBEV8rJLsYzabhSBWcRHBFxZGSKFYmcIyhZYSi5hOwRSKqFEQQREkhICkVBCxCNgIqQWbgIoPEIIQ1ELR9fu9Z8LsZLOJhSByf/iZuec/c+45M2fnbpJkyPAfwTm3F36AdeOLrq6urbGfR6lU6sfnh/lqfJrP57cwDsFvQZw6vrdZsob52UjTuouKVygUNjN/HmrFYvFI9NqlwPEKfAffkkAh1gVLbAouwGlM7aGOrRO+hi+r1Wou1BRTseGz3t7eDaEG2rDfhMPM10TaUuRyufU4T5TL5XHGL+xMLfYBbein4Sg+Py14A1i3CfusJd0ZakHCs/ILNXa5qtNgXBfalwXOFQLdYTzh0qM8Gvtg77OEzzH/js/+2McKn1Fi0SmpWBVab1KMtAtwT2BrDXrmmAIy7ibYVxftnipHP8+YR3sE57q7u7eHPgI+HaY3nBLzndgmXNqrDQnLTwkzbfO2FaFkWHiYQDvgPByL9EHplvAr16R/PfC9HyXczvNlCjzo0t1f1Gq12lr8x9USQYjW8P2rI9SuMZ+Dk4lVTMAyLxhh2m5FNe1fD0u4Dgf0zKkdwDaapEffUIyd7Jlw/Yrw/cvCjqAHZzRP0iRHlLT5jrhl+tfDfH7/DtRKjNe5JovSwmLsxrnX09OzLY7REr5/7VHXy6SzHtWOog1KUEGuRf96+IThSZF1p7zG8w1fDOMQ2vFw7apQtv71zwQag/PYD8FL/qoJrqVl+1ewRJXUNcZblUplo9eCYoa186u+xjysBabUFt6mYC69KaZhn7erKLdC/wq2e0rqI/P+SPMJL6i3Q21VIMA+Fj8Mvzz+hdYmi1eNS3e+Zf8K+AzY+ge6BUItKOZu0uKUlkDVseiTLRb1nR+SVkrvxSf6UegZ+1X4OfTF57H+AzRGTaH18I3GJpoSfv9H19jfhgrRByhp8iGwPzq7YnuGDBkyZPi38QsD7fEQX5WPDgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAaCAYAAAAkJwuaAAAFAUlEQVR4Xu1XXYhVVRQ+l6vRf0pOg/Nz1rkzQ4NjkDAmCBopPjiYIhYkjEJYNAiGoYyWRfmgDypoZCDapPhkwqAPZonJpNiDSQ8VmEEEJqKgmCAmaIz2fXevfWbdPffOvYo5L+eDxT57/ey99tprr71PFGXIkCFDhpogIrNAV0B3SXEcH2lqanrMy9va2p5OkuSolysdqK+vf8KO8zCB+d8C/R34dAk0CLrNNRQKhZehmgtt/2/k4MAuOgG6BZoeKoC3CNRvgzya4EbCn2MMIALX4vn0D7wNGtTXrE0taG1tfQ52AxhzaSirChiNR9btRbtS3O5+HgW7Cd4qULfljSYaGxub4M8FBjM8HVhHJ/j/lJNVA2zniMvqOaGsKnAMXoThNgwwEXRWHSwYlTHof0E9wxtV6ILvgDaGMnGn5y7oMBLk0VA+EmCzToIsrxkw7MaEPfxGu16dWOHlDQ0NEzRjxw9ZpbJm6Haxllo+F1BXV/ek71NOPeqz39nZOba5uXkq9BZ6HpBraWl5HvO8zpZ9bx8COmsZyDKZw03fBxrE+AsCWdEv0CvBvHkeaSQKzOQ70PegAnTGlRhXA4y28jjwG5O/gP410Cks5hnyIJuB/vZSK7fzDDBoE75/wXF7lnw6i/5BtPvRzTOg+N6JcT4GnQcthfwQ2nfQfgi6gu9XxdXpj6D7JtpzoNXBlEXo+IclyBweY/Q/Af8qdN6IzEZw48DvBf2s83bTZ86BgIJVPJEHxWX5T/SFenaMERFrfWTWKcvv6B3w55KBtocTGzNflHczeOKOw0XuKGVcnLhbdJXaz+UYLA3gXQcd85skrpz8BboM+TQ/Pn0C77jNag9cKI3iAn0dc52gHjdIff4grIsMIuQ7ID/rfSTElYBrTB724wdRHyMTeS6cToH26S04rD7yWIL/NgOC9hR1wR5DGb67QLcw7gwdb7kGkY7/C5rlx8H3JNBV6K71PAYPvOOg/kjHtNAFl9RHDdYecdk92errvIPgL7F89OeLe/LNV70HUx89THB4xGczO+Iy9ZGAfDroJnQWG95G0DlmTqD7GeiMyX5uyAIxQVe9YnDF1GkLBl1cHe8K+MXAiJ4EIqlQBlTGk1YMpNG75wuK4PtxOxbzUijg7qlTv2HgTaHcQxeVOlnJofb29qfAOylBlokLbknQ0V8h5shZjBQYcRtIn9MyJEOlIwwQS1i/6AvFPKeGvQKqglnGbLMZ4oGjWC/uKVTiWAjai6llXJy4+shjwkVs7ujoeESGsizNlnJHmKUEYxwhaVnptbWz0vvRBDg9qoSu40/66XkEdCaLKwOfopuLXbngI76Y5djEmeC9a20qAkazQQcQyMdDGZG4pxAXPymUeUDWBzrJjIucQ2v8YhJXa5epXrE+xuWPcBpcPnu4QNj26NOqz144Ouaw9yP44zD2j37uyGUcTxtfIX1+Y6irt/tX4H/rLz3OJ5rl5KHdYS+mstDo3+CkSvwtnBfq0Qk6EFeoj4ReOsyQfjizH/Q+vgfQnkZ7iLc79XRT0icSIe5Suinml1Sz9JvE/d9/jXYK+bF7Ml1Wfz39EeuzLTKbCNoCu/fEBTunWfyDuNLyJWS/ou21v7u6gb+LewKl8z5U8MbkETKZk2e5IN/rJMEDXZHXTQrfasPsa0QucViomZ+3Mm6ibqzlp+B8lN/HvBkyZMiQIUOGDBky1Iz/ALY8plJ27GonAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAaCAYAAADWm14/AAAB7UlEQVR4Xu1UPUsDQRBNCsFvLYxHcl/JJRDUSq4S1EqLFFrYam+v+EFaSSVYiJVgIUGsLAOiFoqFFoL+AUFFsJJUFhpifJPshsl60bMRhXvwuLt5MzuzO3sTCgX4D7BtOwpm4/F4r6r9BsJIvgY+6rpuqCIhFov1WZZ1Bp8yWBF8oxhhK5MODqqx3wJBLhZ4IdK7qnPAJyOS57g9lUp1w1YAi+Aw176EYRhtCMgj8QOeJTxHVR8O+GRFARlVQ+yK0DZVrSngPAOuo/erFIxFplQfCfi0il0+wc9RddhzooCG02mKZDLZD+cD9Ndk1c+qfhI4LR36HXiiaVoH1xzH6YH9EnwFR7jWFHBcQuI5eqedUwFUiOonAW0CPu8eOwxDWxbaIn0r+mcgYAjO+Ugk0im+qwV4LF6HPCW04gjPbcZr8CqRSIyH/CR3XbcFAVumaY5JG10+2EpYfJf7SrD+FynOrs2NKn88O5Bs0m78nzkLlEyNYf0/T6fTXaruG+Ky7GPBFLfTTsB78FS2hYP13/8v5gUssIAdznvYZQG36KXmoVf/fxz/tKr5Bd1U6vMNTsFSRXEyF6KIKNdoWCH2EPZncIBrviAuDY3IiiANkvrMxveGXZvrUqf3Hbt2Ksce2h5mRzvPESBAgAB/Gh/McZn6imdp9AAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAaCAYAAADBuc72AAAC2ElEQVR4Xu2VO2hUQRSGd3EFxRc+4pJ9zb5gWbRytRC0ULSwiIR0EqxSaCGCQgwKVmJtXLsgiIWEaCxtNIQFCwOCWAiKICqoQUVtksIEjd9x55rDuXdd0cbi/vCzd8//z9wzM2fOTSRixPg3NBqNlYVCYVuxWOyHRUJJieXz+Yr1RsE5V2fYKbjKaiFg3gdn4ZLiZ/jBP8+RzGi1Wl1vxu2HL2ETDsJb8DYvnYDHtDcKsiB8Nxlzv1arrbN6RzDgKlwkqT0mvkOSZtK7PT09ayXGjh3G91x2U1mT+C7AeeINFY8EvgH4Hb6GvVaPhKxIVgZflEqltNYkOeItmZQEDpTL5Q08z8AT2idgAbuIP8hkMlusplGpVLbim4bv4Szzlq0nEpjr8BOc5G9Ka0yykfhD53dbKM9wQPu8t8HOX+YxaTUNxg7jOw6vuz88gZ+Qo3TtejxtNWK74Vc4I7vJpH3eO53L5TZpr/yHVR2z8JdvQk7KJxoqt45w7QsRGuCPeQp+ZDE7vbcE3/hkhc/gxW4JeqRkx5lrr/zhfSMyhyzeGkNQNSi7dgOOCf1q3zHJtWw2m9NjiB9yy10h4EK3F6IfhKMJX15BonDQWMNwy/V5j4vEj+sN2K2/ycXDNwSf+Bfe6TTGn8643nlJUMZJwtobCVWf56xmITubTqfX2LhP+ClsBS3MQi6Pf08Um9Yfgpicbz1WM0jhu1KIaCWqfEJdQ0CSRbRxFrpZx+W2E5+XMtPxEFT/fMWRZK2uIbpr9896hFaVOVxEywIpqUupTysEicKpqJP6BY59O6Yv7je1FUA6Ar5FfGcTqk+S5GrX/qqNyWdRDRGswH8ErSU1ajSp2wLaW9EjS0bag2t/upYU5fs+ZL0B0IZJ9hK/k/ARz2fgSdeuzabdEbSjxBfU/I+DL5Z4pZcS+6b0OXhez/FX4MLU/I4lpeZAP8n02U9ujBgxYsT4f/ADTsLo7i6MEWIAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAaCAYAAACtv5zzAAABzUlEQVR4Xu2Uu0sDQRDGExJB8VGoMZjXJpdAEBSEoJUWghYW2qQUK0F7RYOPwkYESxtBU1uIYG8lKIJ/gFhZCKIQECstlBB/w+3J5hLQJBYifvBxuzOz3+zMzZ3H88vhU0pl4T5cS6VSHe6AhhCPxxdisdiIx060AS/C4XCXO64uBAKBNgTPYF72JLJYP8BJd+wncI7BR1gyWIBvsIjIFc8soT6Jj0QiA4lEQumzfZKAmPEy0WogMA/fdfkOfOzndaIV9l7DJxXkaNmpVGbaK5BOp9sROYe33C5o+rD1wju3j/UwtkMq6jTjq0KX+gSP2fpNXzQaHcL+Cq9DoVC32Lj1ILffDgaDrZIgmUz2mGcqgMg0AiWZELcP26b44JLspf9gx7KsmFRHolm5hPtcGQjcVa7+ZzKZJmxzurJl2ZOsmfWJTujwnjGNmHplMEZPpuZSr2/kMAn3Gp5xVb3/Xm67quzpmTDCa4fTf7ho2hHOYHtR+qOqG6pK/7V9RifeMu014Yv5l8TyHnKmvSbQnn5EnlXl/Pt5B0dmAtbr3/olCBAeVfbXKS1wWJD34cQo+5dclEQy66wP+KhaTJ2GIW1DfEom6cfF//GPP44P1nqDMcKj3F8AAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAACvUlEQVR4Xu2WzYuNURzH79NQ4y2J223u23nuTe5Gojua0iylLGjCQiwtWEwWRJZK0iALmSEaJUmysPEyRc2Ekplix8ICpfwFSiQ+P+ccnTnzPPM89z535/nWt3vO93zP7/zO630KhRw5cvy3aDQaLaXUeXgdHqhWq8t8TwyCMAyH6HMZTtTr9V3tdnupb8oQv3dg4L3wHQlvLhaLKymfgU+bzeZq3+shwHcCjjMRJQQX4GO3b1fxMfcLfb1blMvlGoN+gAetxq6soT4HR12vj1qtthHPg1artcpqslMyUXLcL/Wu42NqymoQaKxSqaz12zuFJAC/EbftyLJTd+CM7ICjz4McUzxv/DzI7Rb6MSlniS8IOCabMD6Bk7DhG9JC6bvlJ2KT/SoL6+ouaBvG8xPvLMdwg2iUQ/TX6FukniX+PJhLfl8oZb89CWbAuEQW6C7k0VH6cfkNf8BL8AXcYz1xceL0RNBJXoJJAjyHQ0iB7/FhHoaZqAHTJsIdXI73npms8L2cNmnrRfxY0HkAjqeZcKlUWoH3WdSAaRIxO3oR3obb4FszWek3nDV+IuRxIMBVAr0kYOi3u4gbME53gecInmn7oMiLi3YK7Rf9pkw9Mk6cngpK7+YVGTxM2E0LvGejBjSJfGHRqq5uYXcL32G/DX0Ufjb5dBU/EkrfzxvwobkfiRO04L9wt9I7sN1qJNGP9kgoZSP3hRp/6879+/f/aEHMreivuL/rOogfi578xcgxZ7BZeNpqHLn1stqh+dMXUD+k9P27S3WJ0Y7LZL0vnD4mdc7GSxs/CkGovy2nlT6mA76hU7Dqg8T5SIIn4T7Kc4wx5n6z1vXHwXfxFMyJkTtI/Rr6J36PmsWQx+emHO1O4i8App2yYv7XSFZIYiS0gwRG5LPNb18M4pd+i/XNEj9Hjhw5cuToAf4Ax5z/ACf+qZMAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAaCAYAAAAe97TpAAADVklEQVR4Xu1WXYiMURieyShCfqfJ/J1vxta0uWBNLvyVxIULmvYCxd1ekORCMdwp7YUS2b3ClqRtw7qzJUnuKOWKSG2WBkWbknWxrPU8871n5p3zfbP2AqW+p57Od973OT/ve857ZmKxCBH+H5TL5bn5fH6153kV0IMpTlsul1vlav8IjDEbwQ/gtPB5Op1e4eossLkd0EyJlu39TCaz3PrR3wa+BvvA/eAt8DZiuQEeVLqdMsdseMyOmxEQngPfgTVsKuv6CW4W/iHwCzgMU0L7kendCPIVT0GZ49CeASdgLyt7HQjsGnzf4dvsuDhuHfgG7HZ8QSSTyYUQXseEF9stBsThPwSeguanm51isbgYtsfgEW0nENx62B+5J4x1lsL+BBzLZrMZ7bOAbwC67a49AIiKEF9BuwftNNpdrgb2LgnipAnJHPu0h2WNSWGC8BnX9kKhsMYETzUB7VrWEDscR11zVBvwGjDDkrFvbpaRpfnwn2a24BsBRzFxSmsYOBMAPoBumfaxD3ZoG2H8mmHSqtYmawykUqkFounG2kuao9qAG+SRYUAnOA72Ov699MsCY07m6oCtANYkEPIl5wnbvAU3C/7A/BW0K5EYg+9L7vq/ha0HFjOzi+9RcDAmR49JPclUQgIN1IOF8V+cjxKE5WTY9VT1wBeuhv5bGTs1qxrQsPWAvc6TgB6S/I75G68yENFWTUg9uJBk9IDPJJARzq81rBPYJ4w61VKptAi6m9yTyObQ1hzVBrYepMtnbdDInWdGeJXo4Ca4GetTU/Dpzdo7rCHBvFBJacBIPRh1qqLvtwGj3Yf+4eaoNvCkHmwfg3rBcdi3gmdZ1LRzo8a/8249JLiwyl4D6mTdMXHor5oZTtVec7Dg+logwiG9AWbG+C/UMNhl7QzUhNSDFDt/Hzq1XXwdxn8IWp5dVQ+BU7WA5gD8/THnWQ4Awk0Q3tH3Li9PpVyxxgTGP6FA5tin3dXzBI3/+ly2b75FWD1Y8Fp6/m/RJ3CD9rUAdbAFgs/crHAS7KGPC2CSe/b/EOznwa9aC81d+1uA/nH0Lxj/5J7i+wR41Pi10KdrRU7zvZqr8TKRnNv6ZI36Vf7rwHUoSab5t4So8DTbXZMIESJEiBDhX+AXISAkCzx7yCkAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAADOklEQVR4Xu1WO2hUURDdxQiKiviJS7Kf+3ZXWFKIhkXEqGChYJBYWKhgaZPCKiAbFfwgFoJERDs/ICEENdj4QWQLe1tthIBKNCBEQdQiYuI5uzPZm9l92dcIgu/AcD9z7tyZe2fue4lEjBj/N4IgOOac67fzfwXYqA8yDZkXedPd3b3R8hS5XG4/OL+Fy7aaTqc3WJ5FJpPZDO4nyBmrU0A35PnRTqIdEIhXIR8hU3A0Y/UEA4B+HPINMoGpDssJQQf4t+kQbu2eVfro7OxcDd5LyDQOsWDUyzB/kD5ms9ntRtcMMTaKTa+j/QGDZcsBktAPQk6DMwcZsoQwyC2/hvxqdyAMhkExOPpl9eLrQ0iP1TVBjN1CewTtPNoBy8F8rwQ2TAfB2W05rUBHsOa+BPc+zGEF9P30AXJZ55DGK2FjG7pJCexuPp9PectaA9d6iDfB68Win87chhi+gDYN3VPIZCTDidqDMQj+KUiXBPYKQa6zPAUDYmD0SeeYQZgbYR/2VqB/mG1jVQjoNBbvw4IeyIx/WqI/Sr0E9s61SScFHqEsuGOsTa92GFyX5RKpVGoVdFXxYSd54tMzyHHLXxJaX3wweAvoT9IZqJLUI6gAQVXQ7ZDgo9YXa/KKnnybR6EGr75m0f9AQf+LBNq+pnxoffFqvc21DhhMhcEJt+Ii1heW7KDdcrm8XKb4Mk648MepVhLO1BcPHONHmr60x5ttrAqB1pcMkzAy5qSGeENMQyoYuItYX6xJ8B6Lk1bmaNeuIZzUl/O+UdhrK/a+6HHOYnxAx6EIpL50LMZnML+XqUQnOS8nN+Ui1BcPAzbPJSSdFXLjLV/dKKkqNTtaKBTWWt0iiLFx35Crf/35MjJtenWewbsI9cUDQGAPisXiJqvTwFrZoA8MClINSTVm0yXISatoAoztAvFJqVRa480NcHNJz4UTd/WbXLK+5FW7A855qyNyje8kH6NFaFVfCvnjueHqL3Le6hcAI3tA+CqGKLOQE9Rh0zKCeqH/f5gfgXz3ueA8R4quV3tSf6Meh85fUz24WzB+a/R88fqC+nfus6fzX0Smvv6XUm4mTHrHiBEjRowYMf5x/AFNXCA5afRCawAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAACsElEQVR4Xu2WT6hMURzH7/QoIpLG9ObfvfOnJCU1yualCDt/ioVYWpBEUWwsXun18mwkKXqSBRYWVpMF5cVGnqxkY4GasleKxPP5vTmH3zvOnbkz91m53/o253zv9/c7v3PmnHNvEGTIkOG/Ra1WWx+G4RS8CQ+Xy+XlrqcHclEUbSXuKrxOrm2iaUPK/IsDBj4A31Ls5nw+v5L2Rfi4Xq+vdr0uCoXCimq1ehv/Q7iB9kZ+X/G7y3rS5E8EKaLVai11dY1isVhh0HfwiNUocg39WXhSez3I4b3CBJ7Zook5A+fQz0s/Zf7eMNvkAbzXbDbz7nMNKQB+YfCWknNod+GM/ANKXwCJkVgmesxqpVKpjDbBs7r00+SPgz0nT+E1WUnX4EPYPVduIQG57qB/sgX7gGccz3c8Y2ZLjqIt0540+V2MELADPoeXWdG1rqEXzIBxhfylW8iEeN42nkn6N8LutpVteiIwl1FcnjjdhxHM+zC/gBfYoqtcQz+Yf2HGN2C/QlTsHJxGWiI67e3wc6VS2ZsmfyCXC6ZDmF5jOi0XjutJCoklzxPfgP0KUZP4iWen1emPwo+wLbtr2Px2xT5gPL4Y76K4AeN0i7hFUhMVypn15onTF8D5V88Ns20tyDHhG9AU0pFbVOsaPJ92Y92JpsmvYc+pvJMGvogEcpaI/aG3X/TnomlL28gyluD3rWoW+xuxY1ZTE52PHSB/IuQi9WqRwVxDHGRxiH0Jx63GkWiSoyMTsRr9o2H34rkfmIun0Wiso/+maj4OjE+O1vxlJP2k+QdFjo+FTSR5BG/BmmvwgaK24H0vxwAepD1LEZf0VxX6HvSv4gnUd2zYfbV10CfhKWnDs9qTJP/QkEnCKVk995kPcrlQ0G4K2J/0Y8MiSWwST4YMGTJkyPAP8Qv6ugKywQbgsgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAaCAYAAAD1wA/qAAACPElEQVR4Xu2VP2gUQRTG90gERVEUz8P7t/cPBcEoLCkEFRQipE8RENuIIChaBPwDFlpYBIIWViIWiQiC2IhFCrGxCAiCIaCkSLBJYSMkTYr4+7y3x9wkxQXJEmE++Nh5896bfe/bmdkoCggICAgI+Hf0VavVpF6vn0+SZJfraDQaB/y5jPC3pkqlcqlQKOzFzlHLMeaGyuXyHj84UpE4J+M4vlOr1T7xnEh9NHYce5n5UTdnuyHxeO80vAXvwq9W4yP4HL6lpt1dSeoQx71Wq7Wf50cCXjPdJx/2ZbiKKoNdSQ7w32eNpS3wA4oe8tdxkGPNh7zznAzGR+EifIWwAzx/qc58Pr+vK4vCrxFwCucZFe2qj/0EzhWLxcNuznZCTUrYVHGr7bdEtd1zBZ7w8zog8QHBP2FdNsEHGc/CKcycF54Z1ABcoZ7E922APpM+F3yD2a+5VAkavOqFd8Fy9fl7YrPZPBLZ1u0BORp4Qd6shPWdG2AvWSR43JmTEmvMnXVjfbAVThIz0itZc3jTW8dgB/0psWPa0ozn1Exku4LxkHxeWhuoXyBhoWqN2GIzWiTL8yGoUbgOJ+AFE7NTFzvkJUK0/LwUuilux+0zoutNZ2PdVSIr8M4G7/4Cpxi/o/CbjH9YXe/hRT+nA90Qova7fR3dYDofmf4/UqgW9yz59qYolUplip6HM/YH7UeJZ9jfLPn/gN1Oy/B61G7iBvyOCqf92B0N/WTi9vn4LNLAY/3l/biAgICAHYE/r8ibzqI70qAAAAAASUVORK5CYII=>