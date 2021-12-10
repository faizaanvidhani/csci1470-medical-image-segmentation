import tensorflow as tf
from preprocess import tensor_to_image

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT_bool = GT == tf.math.reduce_max(GT)
    boolarr = SR == GT_bool

    #Displaying Image
    binaryarr = tf.where(boolarr, 0, 1)
    binaryarr = tf.math.scalar_mul(255, binaryarr)

    SR_image = tensor_to_image(binaryarr[0])
    GT_image = tensor_to_image(GT[0])
    SR_image.show()
    GT_image.show()

    corr = tf.math.count_nonzero(boolarr)
    tensor_size = SR.shape[0]*SR.shape[1]*SR.shape[2]*SR.shape[3]
    acc = float(corr)/float(tensor_size)

    return acc


# NOTE: Functions below are not used. Future work should consider evaluating the model by these metrics.

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == tf.math.reduce_max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = (tf.math.count_nonzero(SR) + tf.math.count_nonzero(GT)) == 2
    TP = ((SR==True)+(GT==True))==2
    FN = ((SR==False)+(GT==True))==2

    SE = float(tf.math.reduce_max(TP))/(float(tf.math.reduce_sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == tf.math.reduce_max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2

    SP = float(tf.math.reduce_sum(TN))/(float(tf.math.reduce_sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == tf.math.reduce_max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2

    PC = float(tf.math.reduce_sum(TP))/(float(tf.math.reduce_sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == tf.math.reduce_max(GT)
    
    Inter = tf.math.reduce_sum((SR+GT)==2)
    Union = tf.math.reduce_sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == tf.math.reduce_max(GT)

    Inter = tf.math.reduce_sum((SR+GT)==2)
    DC = float(2*Inter)/(float(tf.math.reduce_sum(SR)+tf.math.reduce_sum(GT)) + 1e-6)

    return DC
