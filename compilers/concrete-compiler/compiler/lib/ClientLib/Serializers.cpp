// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <cstdint>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <optional>
#include <stdlib.h>
#include <variant>
#include <vector>

#include "compress_lwe/defines.h"
#include "compress_lwe/serialize.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

template <typename Key>
std::ostream &writeUInt64KeyBuffer(std::ostream &ostream, Key &buffer) {
  writeSize(ostream, (uint64_t)buffer.size());
  ostream.write((const char *)buffer.buffer(),
                buffer.size() * sizeof(uint64_t));
  assert(ostream.good());
  return ostream;
}

std::istream &operator>>(std::istream &istream,
                         std::shared_ptr<std::vector<uint64_t>> &vec) {
  // TODO assertion on size?
  uint64_t size;
  readSize(istream, size);
  vec->resize(size);
  istream.read((char *)vec->data(), size * sizeof(uint64_t));
  assert(istream.good());
  return istream;
}

// LweSecretKey ////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweSecretKeyParam param) {
  writeWord(ostream, param.dimension);
  return ostream;
}

std::istream &operator>>(std::istream &istream, LweSecretKeyParam &param) {
  readWord(istream, param.dimension);
  return istream;
}

// LweSecretKey /////////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweSecretKey &key) {
  ostream << key.parameters();
  writeUInt64KeyBuffer(ostream, key);
  return ostream;
}

LweSecretKey readLweSecretKey(std::istream &istream) {
  LweSecretKeyParam param;
  istream >> param;
  auto buffer = std::make_shared<std::vector<uint64_t>>();
  istream >> buffer;
  return LweSecretKey(buffer, param);
}

// KeyswitchKeyParam ////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const KeyswitchKeyParam param) {
  // TODO keys id
  writeWord(ostream, param.level);
  writeWord(ostream, param.baseLog);
  writeWord(ostream, param.variance);
  return ostream;
}

std::istream &operator>>(std::istream &istream, KeyswitchKeyParam &param) {
  // TODO keys id
  param.outputSecretKeyID = 1234;
  param.inputSecretKeyID = 1234;
  readWord(istream, param.level);
  readWord(istream, param.baseLog);
  readWord(istream, param.variance);
  return istream;
}

// LweKeyswitchKey //////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweKeyswitchKey &key) {
  ostream << key.parameters();
  writeUInt64KeyBuffer(ostream, key);
  return ostream;
}

LweKeyswitchKey readLweKeyswitchKey(std::istream &istream) {
  KeyswitchKeyParam param;
  istream >> param;
  auto buffer = std::make_shared<std::vector<uint64_t>>();
  istream >> buffer;
  return LweKeyswitchKey(buffer, param);
}

// BootstrapKeyParam ////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const BootstrapKeyParam param) {
  // TODO keys id
  writeWord(ostream, param.level);
  writeWord(ostream, param.baseLog);
  writeWord(ostream, param.glweDimension);
  writeWord(ostream, param.variance);
  writeWord(ostream, param.polynomialSize);
  writeWord(ostream, param.inputLweDimension);
  return ostream;
}

std::istream &operator>>(std::istream &istream, BootstrapKeyParam &param) {
  // TODO keys id
  readWord(istream, param.level);
  readWord(istream, param.baseLog);
  readWord(istream, param.glweDimension);
  readWord(istream, param.variance);
  readWord(istream, param.polynomialSize);
  readWord(istream, param.inputLweDimension);
  return istream;
}

// LweBootstrapKey //////////////////////////////

std::ostream &operator<<(std::ostream &ostream, const LweBootstrapKey &key) {
  ostream << key.parameters();
  writeUInt64KeyBuffer(ostream, key);
  return ostream;
}

LweBootstrapKey readLweBootstrapKey(std::istream &istream) {
  BootstrapKeyParam param;
  istream >> param;
  auto buffer = std::make_shared<std::vector<uint64_t>>();
  istream >> buffer;
  return LweBootstrapKey(buffer, param);
}

// PackingKeyswitchKeyParam ////////////////////////////

std::ostream &operator<<(std::ostream &ostream,
                         const PackingKeyswitchKeyParam param) {

  // TODO keys id
  writeWord(ostream, param.level);
  writeWord(ostream, param.baseLog);
  writeWord(ostream, param.glweDimension);
  writeWord(ostream, param.polynomialSize);
  writeWord(ostream, param.inputLweDimension);
  writeWord(ostream, param.variance);

  return ostream;
}

std::istream &operator>>(std::istream &istream,
                         PackingKeyswitchKeyParam &param) {

  // TODO keys id
  param.outputSecretKeyID = 1234;
  param.inputSecretKeyID = 1234;
  readWord(istream, param.level);
  readWord(istream, param.baseLog);
  readWord(istream, param.glweDimension);
  readWord(istream, param.polynomialSize);
  readWord(istream, param.inputLweDimension);
  readWord(istream, param.variance);

  return istream;
}

// PackingKeyswitchKey //////////////////////////////

std::ostream &operator<<(std::ostream &ostream,
                         const PackingKeyswitchKey &key) {
  ostream << key.parameters();
  writeUInt64KeyBuffer(ostream, key);
  return ostream;
}

PackingKeyswitchKey readPackingKeyswitchKey(std::istream &istream) {
  PackingKeyswitchKeyParam param;
  istream >> param;
  auto buffer = std::make_shared<std::vector<uint64_t>>();
  istream >> buffer;
  auto b = PackingKeyswitchKey(buffer, param);

  return b;
}

// PaiKeyParam ////////////////////////////

#ifdef OUTPUT_COMPRESSION_SUPPORT
std::ostream &operator<<(std::ostream &ostream, const PaiKeyParam param) {
  writeWord(ostream, param.secretKeyID);

  return ostream;
}

std::istream &operator>>(std::istream &istream, PaiKeyParam &param) {
  readWord(istream, param.secretKeyID);

  return istream;
}

std::ostream &operator<<(std::ostream &ostream, const comp::FullKeys &key) {
  comp::writeFullKeys(ostream, key);

  return ostream;
}

comp::FullKeys readFullKey(std::istream &istream) {
  comp::FullKeys keys;

  comp::readFullKeys(istream, keys);
  return keys;
}

#endif

// KeySet ////////////////////////////////

std::unique_ptr<KeySet> readKeySet(std::istream &istream) {
  uint64_t nbKey;

  readSize(istream, nbKey);
  std::vector<LweSecretKey> secretKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    secretKeys.push_back(readLweSecretKey(istream));
  }

  readSize(istream, nbKey);
  std::vector<LweBootstrapKey> bootstrapKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    bootstrapKeys.push_back(readLweBootstrapKey(istream));
  }

  readSize(istream, nbKey);
  std::vector<LweKeyswitchKey> keyswitchKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    keyswitchKeys.push_back(readLweKeyswitchKey(istream));
  }

  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
  readSize(istream, nbKey);
  for (uint64_t i = 0; i < nbKey; i++) {
    packingKeyswitchKeys.push_back(readPackingKeyswitchKey(istream));
  }

#ifdef OUTPUT_COMPRESSION_SUPPORT
  std::optional<comp::FullKeys> fullKeys;
  readSize(istream, nbKey);
  assert(nbKey <= 1);
  if (nbKey == 1) {
    fullKeys = readFullKey(istream);
  }
#endif

  std::string clientParametersString;
  istream >> clientParametersString;
  auto clientParameters =
      llvm::json::parse<ClientParameters>(clientParametersString);

  if (!clientParameters) {
    return std::unique_ptr<KeySet>(nullptr);
  }

  auto csprng = ConcreteCSPRNG(0);
  auto keySet =
      KeySet::fromKeys(clientParameters.get(), secretKeys, bootstrapKeys,
                       keyswitchKeys, packingKeyswitchKeys,
#ifdef OUTPUT_COMPRESSION_SUPPORT
                       std::move(fullKeys),
#endif
                       std::move(csprng));

  return std::move(keySet.value());
}

std::ostream &operator<<(std::ostream &ostream, const KeySet &keySet) {
  auto secretKeys = keySet.getSecretKeys();
  writeSize(ostream, secretKeys.size());
  for (auto sk : secretKeys) {
    ostream << sk;
  }

  auto bootstrapKeys = keySet.getBootstrapKeys();
  writeSize(ostream, bootstrapKeys.size());
  for (auto bsk : bootstrapKeys) {
    ostream << bsk;
  }

  auto keyswitchKeys = keySet.getKeyswitchKeys();
  writeSize(ostream, keyswitchKeys.size());
  for (auto ksk : keyswitchKeys) {
    ostream << ksk;
  }

  auto packingKeyswitchKeys = keySet.getPackingKeyswitchKeys();
  writeSize(ostream, packingKeyswitchKeys.size());
  for (auto pksk : packingKeyswitchKeys) {
    ostream << pksk;
  }

#ifdef OUTPUT_COMPRESSION_SUPPORT

  auto fullKey = keySet.getFullKey();
  if (fullKey.has_value()) {
    writeSize<uint64_t>(ostream, 1);
    ostream << *fullKey;
  } else {
    writeSize<uint64_t>(ostream, 0);
  }
#endif

  auto clientParametersJson = llvm::json::Value(keySet.clientParameters());
  std::string clientParametersString;
  llvm::raw_string_ostream clientParametersStringBuffer(clientParametersString);
  clientParametersStringBuffer << clientParametersJson;
  ostream << clientParametersString;

  assert(ostream.good());
  return ostream;
}

// EvaluationKey ////////////////////////////////

EvaluationKeys readEvaluationKeys(std::istream &istream) {
  uint64_t nbKey;
  readSize(istream, nbKey);
  std::vector<LweBootstrapKey> bootstrapKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    bootstrapKeys.push_back(readLweBootstrapKey(istream));
  }
  readSize(istream, nbKey);
  std::vector<LweKeyswitchKey> keyswitchKeys;
  for (uint64_t i = 0; i < nbKey; i++) {
    keyswitchKeys.push_back(readLweKeyswitchKey(istream));
  }
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
  readSize(istream, nbKey);
  for (uint64_t i = 0; i < nbKey; i++) {
    packingKeyswitchKeys.push_back(readPackingKeyswitchKey(istream));
  }

#ifdef OUTPUT_COMPRESSION_SUPPORT
  std::optional<comp::CompressionKey> compKey;

  readSize(istream, nbKey);
  assert(nbKey <= 1);
  if (nbKey == 1) {

    comp::CompressionKey a;

    comp::readCompKeys(istream, a);

    compKey = a;
  }
#endif

  return EvaluationKeys(keyswitchKeys, bootstrapKeys, packingKeyswitchKeys

#ifdef OUTPUT_COMPRESSION_SUPPORT
                        ,
                        compKey
#endif
  );
}

std::ostream &operator<<(std::ostream &ostream,
                         const EvaluationKeys &evaluationKeys) {
  auto bootstrapKeys = evaluationKeys.getBootstrapKeys();
  writeSize(ostream, bootstrapKeys.size());
  for (auto bsk : bootstrapKeys) {
    ostream << bsk;
  }
  auto keyswitchKeys = evaluationKeys.getKeyswitchKeys();
  writeSize(ostream, keyswitchKeys.size());
  for (auto ksk : keyswitchKeys) {
    ostream << ksk;
  }
  auto packingKeyswitchKeys = evaluationKeys.getPackingKeyswitchKeys();
  writeSize(ostream, packingKeyswitchKeys.size());
  for (auto pksk : packingKeyswitchKeys) {
    ostream << pksk;
  }

#ifdef OUTPUT_COMPRESSION_SUPPORT
  auto compKey = evaluationKeys.getCompressionKey();
  if (compKey.has_value()) {

    writeSize<uint64_t>(ostream, 1);
    comp::writeCompKeys(ostream, *compKey);
  } else {
    writeSize<uint64_t>(ostream, 0);
  }
#endif

  assert(ostream.good());
  return ostream;
}

// TensorData ///////////////////////////////////

template <typename T>
std::ostream &serializeScalarDataRaw(T value, std::ostream &ostream) {
  writeWord<uint64_t>(ostream, sizeof(T) * 8);
  writeWord<uint8_t>(ostream, std::is_signed<T>());
  writeWord<T>(ostream, value);
  return ostream;
}

std::ostream &serializeScalarData(const ScalarData &sd, std::ostream &ostream) {
  switch (sd.getType()) {
  case ElementType::u64:
    return serializeScalarDataRaw<uint64_t>(sd.getValue<uint64_t>(), ostream);
  case ElementType::i64:
    return serializeScalarDataRaw<int64_t>(sd.getValue<int64_t>(), ostream);
  case ElementType::u32:
    return serializeScalarDataRaw<uint32_t>(sd.getValue<uint32_t>(), ostream);
  case ElementType::i32:
    return serializeScalarDataRaw<int32_t>(sd.getValue<int32_t>(), ostream);
  case ElementType::u16:
    return serializeScalarDataRaw<uint16_t>(sd.getValue<uint16_t>(), ostream);
  case ElementType::i16:
    return serializeScalarDataRaw<int16_t>(sd.getValue<int16_t>(), ostream);
  case ElementType::u8:
    return serializeScalarDataRaw<uint8_t>(sd.getValue<uint8_t>(), ostream);
  case ElementType::i8:
    return serializeScalarDataRaw<int8_t>(sd.getValue<int8_t>(), ostream);
  }

  return ostream;
}

template <typename T> ScalarData unserializeScalarValue(std::istream &istream) {
  T value;
  readWord(istream, value);
  return ScalarData(value);
}

outcome::checked<ScalarData, StringError>
unserializeScalarData(std::istream &istream) {
  uint64_t scalarWidth;
  readWord(istream, scalarWidth);

  switch (scalarWidth) {
  case 64:
  case 32:
  case 16:
  case 8:
    break;
  default:
    return StringError("Scalar width must be either 64, 32, 16 or 8, but got ")
           << scalarWidth;
  }

  uint8_t scalarSignedness;
  readWord(istream, scalarSignedness);

  if (scalarSignedness != 0 && scalarSignedness != 1) {
    return StringError("Numerical value for scalar signedness must be either "
                       "0 or 1, but got ")
           << scalarSignedness;
  }

  switch (scalarWidth) {
  case 64:
    return (scalarSignedness) ? unserializeScalarValue<int64_t>(istream)
                              : unserializeScalarValue<uint64_t>(istream);
  case 32:
    return (scalarSignedness) ? unserializeScalarValue<int32_t>(istream)
                              : unserializeScalarValue<uint32_t>(istream);
  case 16:
    return (scalarSignedness) ? unserializeScalarValue<int16_t>(istream)
                              : unserializeScalarValue<uint16_t>(istream);
  case 8:
    return (scalarSignedness) ? unserializeScalarValue<int8_t>(istream)
                              : unserializeScalarValue<uint8_t>(istream);
  }

  assert(false && "Unhandled scalar type");
}

template <typename T>
static std::istream &unserializeTensorDataElements(TensorData &values_and_sizes,
                                                   std::istream &istream) {
  // getElementPointer is not valid if the tensor contains no data
  if (values_and_sizes.getNumElements() > 0) {
    readWords(istream, values_and_sizes.getElementPointer<T>(0),
              values_and_sizes.getNumElements());
  }

  return istream;
}

std::ostream &serializeTensorData(const TensorData &values_and_sizes,
                                  std::ostream &ostream) {
  switch (values_and_sizes.getElementType()) {
  case ElementType::u64:
    return serializeTensorDataRaw<uint64_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint64_t>(), ostream);
  case ElementType::i64:
    return serializeTensorDataRaw<int64_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int64_t>(), ostream);
  case ElementType::u32:
    return serializeTensorDataRaw<uint32_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint32_t>(), ostream);
  case ElementType::i32:
    return serializeTensorDataRaw<int32_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int32_t>(), ostream);
  case ElementType::u16:
    return serializeTensorDataRaw<uint16_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint16_t>(), ostream);
  case ElementType::i16:
    return serializeTensorDataRaw<int16_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int16_t>(), ostream);
  case ElementType::u8:
    return serializeTensorDataRaw<uint8_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<uint8_t>(), ostream);
  case ElementType::i8:
    return serializeTensorDataRaw<int8_t>(
        values_and_sizes.getDimensions(),
        values_and_sizes.getElements<int8_t>(), ostream);
  }

  assert(false && "Unhandled element type");
}

outcome::checked<TensorData, StringError>
unserializeTensorData(std::istream &istream) {

  if (incorrectMode(istream)) {
    return StringError("Stream is in incorrect mode");
  }

  uint64_t numDimensions;
  readWord(istream, numDimensions);

  std::vector<size_t> dims;

  for (uint64_t i = 0; i < numDimensions; i++) {
    int64_t dimSize;
    readWord(istream, dimSize);
    dims.push_back(dimSize);
  }

  uint64_t elementWidth;
  readWord(istream, elementWidth);

  switch (elementWidth) {
  case 64:
  case 32:
  case 16:
  case 8:
    break;
  default:
    return StringError("Element width must be either 64, 32, 16 or 8, but got ")
           << elementWidth;
  }

  uint8_t elementSignedness;
  readWord(istream, elementSignedness);

  if (elementSignedness != 0 && elementSignedness != 1) {
    return StringError("Numerical value for element signedness must be either "
                       "0 or 1, but got ")
           << elementSignedness;
  }

  TensorData result(dims, elementWidth, elementSignedness == 1);

  switch (result.getElementType()) {
  case ElementType::u64:
    unserializeTensorDataElements<uint64_t>(result, istream);
    break;
  case ElementType::i64:
    unserializeTensorDataElements<int64_t>(result, istream);
    break;
  case ElementType::u32:
    unserializeTensorDataElements<uint32_t>(result, istream);
    break;
  case ElementType::i32:
    unserializeTensorDataElements<int32_t>(result, istream);
    break;
  case ElementType::u16:
    unserializeTensorDataElements<uint16_t>(result, istream);
    break;
  case ElementType::i16:
    unserializeTensorDataElements<int16_t>(result, istream);
    break;
  case ElementType::u8:
    unserializeTensorDataElements<uint8_t>(result, istream);
    break;
  case ElementType::i8:
    unserializeTensorDataElements<int8_t>(result, istream);
    break;
  }

  return std::move(result);
}

std::ostream &serializeScalarOrTensorData(const ScalarOrTensorData &sotd,
                                          std::ostream &ostream) {
  writeWord<uint8_t>(ostream, std::holds_alternative<TensorData>(sotd));
  if (std::holds_alternative<TensorData>(sotd))
    return serializeTensorData(std::get<TensorData>(sotd), ostream);
  else
    return serializeScalarData(std::get<ScalarData>(sotd), ostream);
}

outcome::checked<ScalarOrTensorData, StringError>
unserializeScalarOrTensorData(std::istream &istream) {
  uint8_t isTensor;
  readWord(istream, isTensor);

  if (isTensor != 0 && isTensor != 1) {
    return StringError("Numerical value indicating whether a data element is a "
                       "tensor must be either 0 or 1, but got ")
           << isTensor;
  }

  if (isTensor) {
    auto tdOrErr = unserializeTensorData(istream);

    if (tdOrErr.has_error())
      return std::move(tdOrErr.error());
    else
      return ScalarOrTensorData(std::move(tdOrErr.value()));
  } else {
    auto tdOrErr = unserializeScalarData(istream);

    if (tdOrErr.has_error())
      return std::move(tdOrErr.error());
    else
      return ScalarOrTensorData(std::move(tdOrErr.value()));
  }
}

std::ostream &serializeScalarOrTensorDataOrCompressed(
    const ScalarOrTensorOrCompressedData &sotd, std::ostream &ostream) {

  if (std::holds_alternative<TensorData>(sotd)) {
    writeWord<uint8_t>(ostream, 0);
    return serializeTensorData(std::get<TensorData>(sotd), ostream);
  } else if (std::holds_alternative<ScalarData>(sotd)) {

    writeWord<uint8_t>(ostream, 1);
    return serializeScalarData(std::get<ScalarData>(sotd), ostream);
  } else {
#ifdef OUTPUT_COMPRESSION_SUPPORT
    assert(std::holds_alternative<std::shared_ptr<comp::CompressedCiphertext>>(
        sotd));

    const comp::CompressedCiphertext &compCt =
        *std::get<std::shared_ptr<comp::CompressedCiphertext>>(sotd);

    writeWord<uint8_t>(ostream, 2);

    comp::writeCompCt(ostream, compCt);

    return ostream;
#else
    exit(1);
#endif
  }
}

outcome::checked<ScalarOrTensorOrCompressedData, StringError>
unserializeScalarOrTensorDataOrCompressed(std::istream &istream) {
  uint8_t type;
  readWord<uint8_t>(istream, type);

  if (type != 0 && type != 1 && type != 2) {
    return StringError("Numerical value indicating whether a data element is a "
                       "tensor must be either 0, 1 or 2, but got ")
           << type;
  }

  if (type == 0) {
    auto tdOrErr = unserializeTensorData(istream);

    if (tdOrErr.has_error())
      return std::move(tdOrErr.error());
    else
      return ScalarOrTensorOrCompressedData(std::move(tdOrErr.value()));
  } else if (type == 1) {
    auto tdOrErr = unserializeScalarData(istream);

    if (tdOrErr.has_error())
      return std::move(tdOrErr.error());
    else
      return ScalarOrTensorOrCompressedData(std::move(tdOrErr.value()));
  } else {
#ifdef OUTPUT_COMPRESSION_SUPPORT

    auto ct = std::make_shared<comp::CompressedCiphertext>();

    comp::readCompCt(istream, *ct);

    return ct;
#else
    exit(1);
#endif
  }
}

std::ostream &
serializeVectorOfScalarOrTensorData(const std::vector<ScalarOrTensorData> &v,
                                    std::ostream &ostream) {
  writeSize<uint64_t>(ostream, v.size());
  for (auto &sotd : v) {
    serializeScalarOrTensorData(sotd, ostream);
    assert(ostream.good());
  }
  return ostream;
}
outcome::checked<std::vector<ScalarOrTensorData>, StringError>
unserializeVectorOfScalarOrTensorData(std::istream &istream) {
  uint64_t nbElt;
  readSize(istream, nbElt);
  std::vector<ScalarOrTensorData> v;
  for (uint64_t i = 0; i < nbElt; i++) {
    OUTCOME_TRY(auto elt, unserializeScalarOrTensorData(istream));
    v.push_back(std::move(elt));
  }
  return v;
}

std::ostream &serializeVectorOfScalarOrTensorDataOrCompressed(
    const std::vector<SharedScalarOrTensorOrCompressedData> &v,
    std::ostream &ostream) {
  writeSize<uint64_t>(ostream, v.size());
  for (auto &sotd : v) {
    serializeScalarOrTensorDataOrCompressed(sotd.get(), ostream);
    assert(ostream.good());
  }
  return ostream;
}
outcome::checked<std::vector<SharedScalarOrTensorOrCompressedData>, StringError>
unserializeVectorOfScalarOrTensorDataOrCompressed(std::istream &istream) {
  uint64_t nbElt;
  readSize(istream, nbElt);
  std::vector<SharedScalarOrTensorOrCompressedData> v;
  for (uint64_t i = 0; i < nbElt; i++) {
    OUTCOME_TRY(auto elt, unserializeScalarOrTensorDataOrCompressed(istream));
    v.push_back(
        std::make_shared<ScalarOrTensorOrCompressedData>(std::move(elt)));
  }
  return v;
}

} // namespace clientlib
} // namespace concretelang
