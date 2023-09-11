# Yichen-Capstone-Project

# Dimension Reduction and Information Loss in Genomic Data

Version I

Yichen Guo, Georgetown Data Science and Analytics, Georgetown Lombardi Cancer Center

Dr. Abhijit Dasgupta, Georgetown Data Science and Analytics


## **Introduction**

The complexity of high-dimensional genomic data poses significant challenges in bioinformatics, often masking critical biological insights. While dimensionality reduction techniques can simplify this data, the key is to compress without losing essential information. Neural networks, particularly autoencoders, offer a promising avenue for this task, and when combined with Bayesian Neural Networks, they can provide both compressed representations and probabilistic predictions, vital for genomic associations.


### **Background information**

Traditional dimensionality reduction techniques, such as PCA, UMAP, and t-SNE, have been instrumental in various scientific domains. However, the advent of deep learning has introduced powerful tools like autoencoders that can learn intricate data representations. Autoencoders, through their encoder-decoder architecture, compress data into a lower-dimensional space and then reconstruct it, aiming to retain as much original information as possible.

On the predictive side, Bayesian Neural Networks (BNNs) extend traditional neural networks by incorporating Bayesian inference. This allows BNNs to provide uncertainty estimates alongside predictions, a feature of paramount importance when predicting genomic associations where the stakes are high, and the data is complex.

The information bottleneck principle, which focuses on retaining the most relevant parts of the data while discarding the noise, can provide a theoretical foundation for optimizing autoencoders in the context of genomic data.


### **Research Questions**



1. How does the performance of autoencoders in dimensionality reduction of genomic data compare to traditional techniques like PCA, t-SNE, and UMAP in terms of information retention and compression?
2. Can autoencoders provide a meaningful lower-dimensional representation of genomic data that retains critical biological insights, especially concerning miRNA-mRNA associations?
3. How effective are Bayesian Neural Networks in predicting genomic associations, such as miRNA-mRNA pairs, from the compressed representations obtained from autoencoders?
4. How does the predictive performance of BNNs compare to other machine learning models when using compressed genomic data?
5. Can the information bottleneck principle be applied to guide and optimize the design of autoencoders for genomic data compression, ensuring minimal loss of critical biological information?


### **Aims**



* **Aim 1: Investigate the efficacy of autoencoders in reducing genomic data dimensions.**
* Hypothesis 1: Autoencoders will outperform traditional dimension reduction methods in preserving genomic data information.
* **Aim 2: Utilize Bayesian Neural Networks (BNN) for genomic data predictions.**
* Hypothesis 2.1: BNNs will effectively predict associations between miRNA-mRNA pairs and disease indicators.
* Hypothesis 2.2: When benchmarked against other models, BNNs will emerge as a superior method for selecting miRNA-mRNA pairs.


## Method


### **Description of the data**

Two primary data sources will be possibly utilized:



1. Liver Tissue Samples: Derived from 64 adult patients at MedStar Georgetown University Hospital. These patients were enrolled at MedStar Georgetown University Hospital under a protocol approved by the Georgetown IRB. All participants provided their informed consent and HIPAA authorization. The patient group consisted of 39 individuals diagnosed with HCC and 25 patients with CIRR. This dataset comprises 3287 genes related to the metabolic enzyme database, aiming to identify traits associated with Hepatocellular carcinoma (HCC).
2. Genome‐wide association studies (GWAS): A public database containing over 28 million SNPs derived from 17 trillion bases of sequence data from 882 undomesticated Populus genotypes. This rich dataset provides a comprehensive view of genetic variations and their potential associations with various traits.


### Research Method



1. Data Preprocessing:
    1. Standardization and Normalization: Ensure the genomic data is consistent and comparable across samples.
    2. Data Splitting: Divide the data into training, validation, and test sets. The training set will be used for model development, the validation set for hyperparameter tuning and model selection, and the test set for final evaluation.
2. Dimensionality Reduction:
    3. Traditional Methods: Apply PCA, t-SNE, and UMAP to the training dataset and evaluate their performance in terms of information retention.
    4. Autoencoders:
        1. Model Development: Design and train an autoencoder on the training dataset to learn a compressed representation of the genomic data.
        2. Evaluation: Compare the autoencoder's performance with traditional methods using the validation set. Metrics could include reconstruction error and the ability to retain meaningful genomic patterns.
3. Predictive Modeling with BNNs
    5. Feature Extraction: Use the encoder part of the trained autoencoder to transform the training, validation, and test datasets into their compressed representations.
    6. Model Development: Train a Bayesian Neural Network on the compressed training data to predict specific genomic associations, such as miRNA-mRNA pairs indicative of diseases.
    7. Evaluation: Assess the BNN's performance on the validation set. Metrics could include prediction accuracy, precision-recall curves, and uncertainty estimates.
4. Benchmarking and Comparative Analysis:
    8. Other Models: Train other predictive models (e.g., SVM, Random Forest) on the compressed data and compare their performance with the BNN.
    9. Interpretability Analysis: Use techniques like SHAP or LIME to interpret the predictions made by the BNN and understand which genomic features (or compressed features) are most influential in the predictions.
5. Final Evaluation:
    10. Use the test set to evaluate the best-performing models (both in terms of dimensionality reduction and prediction) to ensure they generalize well to unseen data.


## **Literature review**


### Relevant literature


#### Traditional dimension reduction method



* The art of using t-SNE for single-cell transcriptomics (D. Kobak, Philipp Berens)
    * Abstract: Single-cell transcriptomics yields ever-growing data sets containing RNA expression levels for thousands of genes from up to millions of cells. Common data analysis pipelines include a dimensionality reduction step for visualising the data in two dimensions, most frequently performed using t-SNE. It excels at revealing local structure in high-dimensional data, but naive applications often suffer from severe shortcomings, e.g. the global structure of the data is not represented accurately. This paper describes how to circumvent such pitfalls and develops a protocol for creating more faithful t-SNE visualisations.
    * https://www.nature.com/articles/s41467-019-13056-x.pdf
* A Comparison for Dimensionality Reduction Methods of Single-Cell RNA-seq Data (Ruizhi Xiang, Wencan Wang, Lei Yang, Shiyuan Wang, Chaohan Xu, Xiaowen Chen)
    * Abstract: This paper evaluates the stability, accuracy, and computing cost of 10 dimensionality reduction methods using simulation datasets and real datasets. The study found that t-SNE yielded the best overall performance, while UMAP exhibited the highest stability and well-preserved original cohesion and separation of cell populations.
    * [https://www.frontiersin.org/articles/10.3389/fgene.2021.646936/pdf](https://www.frontiersin.org/articles/10.3389/fgene.2021.646936/pdf)
* Evaluation of UMAP as an alternative to t-SNE for single-cell data (E. Becht, C. Dutertre, Immanuel Kwok, L. Ng, F. Ginhoux, E. Newell)
    * Abstract: Uniform Manifold Approximation and Projection (UMAP) is a recently-published non-linear dimensionality reduction technique. This paper comments on the usefulness of UMAP in high-dimensional cytometry and single-cell RNA sequencing, highlighting its advantages over t-SNE.
    * [https://www.biorxiv.org/content/biorxiv/early/2018/04/10/298430.full.pdf](https://www.biorxiv.org/content/biorxiv/early/2018/04/10/298430.full.pdf)


#### Autoencoders, Bayesian neural networks



* Approximate Bayesian neural networks in genomic prediction (P. Waldmann)
    * Abstract: This study presents a neural network model for genomic SNP data. It shows that regularization using weight decay and dropout results in an approximate Bayesian model that can be used for model averaged posterior predictions. The model is shown to yield better prediction accuracy than other methods and is suitable for both genomic prediction and genome-wide association studies.
    * [https://dx.doi.org/10.1186/s12711-018-0439-1](https://dx.doi.org/10.1186/s12711-018-0439-1)
* A Combination of Multilayer Perceptron, Radial Basis Function Artificial Neural Networks and Machine Learning Image Segmentation for the Dimension Reduction and the Prognosis Assessment of Diffuse Large B-Cell Lymphoma (J. Carreras, Y. Kikuti, M. Miyaoka, and others.)
    * Abstract: This paper employs a dimension reduction algorithm to correlate with the overall survival and other clinicopathological variables in the prognosis of diffuse large B-cell lymphoma. The study uses a combination of various artificial neural networks and machine learning techniques to reduce the dimensionality of gene expression data and build predictive models.
    * [https://dx.doi.org/10.3390/AI2010008](https://dx.doi.org/10.3390/AI2010008)


## Results

What to expect:



* Autoencoder is a useful or significant method for reducing the dimension of genomic data compared to other traditional methods.
* Bayesian neural networks implemented with Rtorch can be used as an effective method in identifying significant miRNA-mRNA pairs for the disease. 
* When benchmarked against other models, BNNs will emerge as a superior method for selecting miRNA-mRNA pairs.


## Conclusion

This capstone project will be mainly focused on two aspects: compressing and preserving information with new technologies such as autoencoders, with the attempts to validate our first hypothesis that autoencoders will outperform traditional dimension reduction methods in preserving genomic data information. The second part of this project will be the attempts to predict significant miRNA-mRNA pairs that could serve as a significant biomarker for diseases like HCC. 