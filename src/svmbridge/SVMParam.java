package svmbridge;

import libsvm.svm_parameter;

/**
 * This just provides a convenient way to set svm_parameter
 * variables from a functional language.  It's recommended
 * that you use SVMBridge.setParams() to create the original
 * object, since it sets defaults to reasonable numbers.
 * 
 * @author rachelgollub
 *
 */
public class SVMParam {

  /**
   * The type of SVM:<ul>
   * <li> 0=C-SVC
   * <li> 1=nu-SVC
   * <li> 2=one class SVM
   * <li> 3=epsilon-SVR
   * <li> 4=nu-SVR
   * </ul>
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_svm_type(svm_parameter param, int val) {
    param.svm_type = val;
    return param;
  }
  
  /**
   * The SVM kernel type:
   * <ul>
   * <li> 0 = linear: u'*v
   * <li> 1 = polynomial: (gamma*u'*v + coef0)^degree
   * <li> 2 = radial basis function: exp(-gamma*|u-v|^2)
   * <li> 3 = sigmoid: tanh(gamma*u'*v + coef0)
   * </ul>
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_kernel_type(svm_parameter param, int val) {
    param.kernel_type = val;
    return param;
  }
  
  /**
   * The degree in the polynomial kernel function.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_degree(svm_parameter param, int val) {
    param.degree = val;
    return param;
  }
  
  /**
   * The gamma parameter in any non-linear kernel function.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_gamma(svm_parameter param, double val) {
    param.gamma = val;
    return param;
  }
  
  /**
   * The coef0 parameter in polynomial or sigmoid kernel functions.
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_coef0(svm_parameter param, double val) {
    param.coef0 = val;
    return param;
  }
  
  /**
   * The C parameter in C-SVC, epsilon-SVR, and nu-SVR.  The optimal
   * value can found through the cross-validation option to training.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_C(svm_parameter param, double val) {
    param.C = val;
    return param;
  }
  
  /**
   * The nu parameter of u-SVC, one-class SVM, and nu-SVR.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_nu(svm_parameter param, double val) {
    param.nu = val;
    return param;
  }
  
  /**
   * Set the tolerance of the termination criteria.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_eps(svm_parameter param, double val) {
    param.eps = val;
    return param;
  }
  
  /**
   * The epsilon value in the loss function in epsilon-SVR.
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_p(svm_parameter param, double val) {
    param.p = val;
    return param;
  }
  
  /**
   * Set the cache memory size in MB.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_cache_size(svm_parameter param, double val) {
    param.cache_size = val;
    return param;
  }
  
  /**
   * Whether to train a model for probability estimates (1=yes, 0=no).
   * This has to be set to 1 to generate precision/recall curves.
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_probability(svm_parameter param, int val) {
    param.probability = val;
    return param;
  }
  
  /**
   * Whether to use shrinking heuristics (1=yes, 0=no).
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_shrinking(svm_parameter param, int val) {
    param.shrinking = val;
    return param;
  }
  
  /**
   * Set the number of weights (length of the weight vector)
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_nr_weight(svm_parameter param, int val) {
    param.nr_weight = val;
    return param;
  }
  
  /**
   * Set labels for the weights assigned.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_weight_label(svm_parameter param, int[] val) {
    param.weight_label = val;
    return param;
  }
  
  /**
   * Assign weights.
   * 
   * @param param
   * @param val
   * @return
   */
  public static svm_parameter set_weight(svm_parameter param, double[] val) {
    param.weight = val;
    return param;
  }
}
